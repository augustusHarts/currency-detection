import { useCallback, useEffect, useRef, useState } from 'react'
import './App.css'

type Box = { x1: number; y1: number; x2: number; y2: number }
type Det = { label: string; confidence: number; box: Box }
type BreakdownRow = { denomination: number; count: number; subtotal: number }
type PredictRes = {
  image_width: number
  image_height: number
  detections: Det[]
  total_amount: number
  breakdown: BreakdownRow[] // Removed the '?' since your backend always sends it
  currency: string          // Add this
  processed_image_url: string // Add this if you want to show the backend's drawn image
}

const API_BASE = (import.meta.env.VITE_API_URL || '/api').replace(/\/$/, '')

function predictUrl(conf: number): string {
  const path = `/api/v1/predict?conf=${encodeURIComponent(String(conf))}`
  if (API_BASE.startsWith('http://') || API_BASE.startsWith('https://')) {
    return `${API_BASE}${path}`
  }
  return new URL(`${API_BASE}${path}`, window.location.origin).href
}

async function predictBlob(blob: Blob, conf = 0.5): Promise<PredictRes> {
  const fd = new FormData()
  fd.append('file', blob, 'frame.jpg')
  const u = predictUrl(conf)
  const r = await fetch(u, { method: 'POST', body: fd })
  if (!r.ok) {
    const t = await r.text()
    throw new Error(t || r.statusText)
  }
  return r.json()
}

export default function App() {
  const [conf, setConf] = useState(0.5)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<PredictRes | null>(null)
  const [imgUrl, setImgUrl] = useState<string | null>(null)
  const [mode, setMode] = useState<'none' | 'upload' | 'cam'>('none')

  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const imgRef = useRef<HTMLImageElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const wrapRef = useRef<HTMLDivElement>(null)
  /** Bumped on each new predict so an older in-flight response cannot overwrite a newer image’s results. */
  const predictEpochRef = useRef(0)

  const drawOverlay = useCallback((res: PredictRes, dispW: number, dispH: number) => {
    const c = canvasRef.current
    if (!c) return
    const ctx = c.getContext('2d')
    if (!ctx) return
    c.width = dispW
    c.height = dispH
    ctx.clearRect(0, 0, dispW, dispH)
    const sx = dispW / res.image_width
    const sy = dispH / res.image_height
    for (const d of res.detections) {
      const { x1, y1, x2, y2 } = d.box
      const X1 = x1 * sx
      const Y1 = y1 * sy
      const X2 = x2 * sx
      const Y2 = y2 * sy
      ctx.strokeStyle = '#24aa4d'
      ctx.lineWidth = 2
      ctx.strokeRect(X1, Y1, X2 - X1, Y2 - Y1)
      const text = `${d.label} ${(d.confidence * 100).toFixed(1)}%`
      ctx.font = '600 13px system-ui'
      const tw = ctx.measureText(text).width + 8
      ctx.fillStyle = 'rgba(36, 170, 77, 0.92)'
      ctx.fillRect(X1, Y1 - 22, tw, 22)
      ctx.fillStyle = '#fff'
      ctx.fillText(text, X1 + 4, Y1 - 6)
    }
  }, [])

  const syncCanvas = useCallback(() => {
    if (!result || !wrapRef.current) return
    const img = imgRef.current
    const vid = videoRef.current
    let w = 0
    let h = 0
    if (mode === 'upload' && img && img.complete) {
      w = img.clientWidth
      h = img.clientHeight
    } else if (mode === 'cam' && vid && vid.videoWidth) {
      w = vid.clientWidth
      h = vid.clientHeight
    }
    if (w && h) drawOverlay(result, w, h)
  }, [result, mode, drawOverlay])

  useEffect(() => {
    window.addEventListener('resize', syncCanvas)
    return () => window.removeEventListener('resize', syncCanvas)
  }, [syncCanvas])

  useEffect(() => {
    if (result) requestAnimationFrame(syncCanvas)
  }, [result, imgUrl, mode, syncCanvas])

  useEffect(() => {
    if (result != null) return
    const c = canvasRef.current
    if (!c) return
    c.width = 0
    c.height = 0
  }, [result])

  const stopCam = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
  }

  const startCam = async () => {
    setError(null)
    predictEpochRef.current += 1
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' },
        audio: false,
      })
      streamRef.current = s
      if (videoRef.current) {
        videoRef.current.srcObject = s
        await videoRef.current.play()
      }
      setMode('cam')
      setImgUrl(null)
      setResult(null)
    } catch (e) {
      setError(String(e))
    }
  }

  const runPredictOnBlob = async (blob: Blob) => {
    const epoch = predictEpochRef.current
    setBusy(true)
    setError(null)
    try {
      const res = await predictBlob(blob, conf)
      if (epoch !== predictEpochRef.current) return
      setResult(res)
    } catch (e) {
      if (epoch !== predictEpochRef.current) return
      setResult(null)
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      if (epoch === predictEpochRef.current) setBusy(false)
    }
  }

  const onFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (!f) return
    predictEpochRef.current += 1
    stopCam()
    setMode('upload')
    setResult(null)
    if (imgUrl) URL.revokeObjectURL(imgUrl)
    setImgUrl(URL.createObjectURL(f))
    await runPredictOnBlob(f)
    e.target.value = ''
  }

  const captureWebcam = async () => {
    const v = videoRef.current
    if (!v || !v.videoWidth) {
      setError('Camera not ready.')
      return
    }
    const c = document.createElement('canvas')
    c.width = v.videoWidth
    c.height = v.videoHeight
    const ctx = c.getContext('2d')
    if (!ctx) return
    ctx.drawImage(v, 0, 0)
    const blob = await new Promise<Blob | null>((r) =>
      c.toBlob((b) => r(b), 'image/jpeg', 0.92),
    )
    if (blob) {
      predictEpochRef.current += 1
      await runPredictOnBlob(blob)
    }
  }

  const reset = () => {
    predictEpochRef.current += 1
    stopCam()
    setMode('none')
    if (imgUrl) URL.revokeObjectURL(imgUrl)
    setImgUrl(null)
    setResult(null)
    setError(null)
    setBusy(false)
  }

  return (
    <div className="app-shell">
      <header className="site-header">
        <img className="header-logo" src="/bargad-logo.png" alt="Bargad" />
      </header>

      <main className="main-area">
        <div className="glass-panel">
          <div className="panel-intro">
            <h1 className="panel-title">Cash detection</h1>
            <p className="hint">
              Upload a photo or use the webcam, then capture a frame to run detection. If a
              blank image still shows notes, raise min confidence on the slider, then upload or
              capture again.
            </p>
          </div>

          <div className="toolbar">
            <label className="file-btn">
              Upload image
              <input type="file" accept="image/*" onChange={onFile} disabled={busy} />
            </label>
            <button type="button" className="ghost" onClick={startCam} disabled={busy}>
              Start webcam
            </button>
            <button
              type="button"
              className="primary"
              onClick={captureWebcam}
              disabled={busy || mode !== 'cam'}
            >
              Capture &amp; analyze
            </button>
            <button type="button" className="ghost" onClick={reset} disabled={busy}>
              Reset
            </button>
          </div>

          <div className="conf-row">
            <label className="conf-control">
              <span className="conf-label">
                Min confidence <strong>{(conf * 100).toFixed(0)}%</strong>
              </span>
              <input
                type="range"
                min={0.2}
                max={0.85}
                step={0.05}
                value={conf}
                onChange={(e) => setConf(Number(e.target.value))}
                disabled={busy}
                aria-valuemin={20}
                aria-valuemax={85}
                aria-valuenow={Math.round(conf * 100)}
                aria-label="Minimum detection confidence"
              />
            </label>
          </div>

          <div className="preview-wrap" ref={wrapRef}>
            {mode === 'upload' && imgUrl && (
              <img ref={imgRef} src={imgUrl} alt="Preview" onLoad={syncCanvas} />
            )}
            {mode === 'cam' && (
              <video ref={videoRef} className="preview-video" playsInline muted />
            )}
            {mode === 'none' && (
              <div className="empty-preview">
                <div className="empty-preview-icon" aria-hidden />
                <p className="empty-preview-title">Add a photo to begin</p>
                <p className="empty-preview-hint">Upload an image or start the webcam, then run analysis.</p>
              </div>
            )}
            {busy && (
              <div className="preview-busy" role="status" aria-live="polite" aria-label="Analyzing image">
                <div className="preview-busy-inner">
                  <div className="spinner" />
                  <p>Analyzing banknotes…</p>
                </div>
              </div>
            )}
            <canvas className="overlay-canvas" ref={canvasRef} />
          </div>

          {error && <p className="err">{error}</p>}

          {result && (
            <section className="results-section" aria-label="Detection results">
              <h2 className="results-heading">Results</h2>
              <div className="stats">
                <div className="stat-chip">
                  <strong>{result.detections.length}</strong>
                  <span>
                    note{result.detections.length !== 1 ? 's' : ''} detected in this image
                  </span>
                </div>
                <div className="total-block">
                  {result.breakdown && result.breakdown.length > 0 && (
                    <ul className="amount-breakdown">
                      {result.breakdown.map((row) => (
                        <li key={row.denomination}>
                          ₹{row.denomination.toLocaleString('en-IN')} × {row.count} = ₹
                          {row.subtotal.toLocaleString('en-IN')}
                        </li>
                      ))}
                    </ul>
                  )}
                  <div className="total-pill">Total : ₹{result.total_amount.toLocaleString('en-IN')}</div>
                </div>
              </div>
              <div className="detection-list-wrap">
                <h3 className="detection-list-title">Each note</h3>
                <ul className="detection-list">
                  {result.detections.map((d, i) => (
                    <li key={i}>
                      <span className="detection-list-label">Note {i + 1}: ₹{d.label}</span>
                      <span className="detection-list-confidence">
                        {(d.confidence * 100).toFixed(1)}% confidence
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            </section>
          )}
        </div>
      </main>

      <footer className="site-footer">
        <img src="/bargad-branding%20(1).svg" alt="" />
      </footer>
    </div>
  )
}
