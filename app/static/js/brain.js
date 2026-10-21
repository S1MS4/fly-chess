import * as THREE from 'three'

// The fly's brain in the panel: every neuron with a known cell body is one point
// at its real position. After each think the points show how differently each neuron
// fired for this board than it usually does, step by step: red = more, blue = less,
// grey = as usual. All 12 steps go to the GPU in one go and the shader picks the step,
// so replaying a think costs nothing. Drag to turn it.

const vertexShader = /* glsl */ `
  attribute vec4 a0;            // steps 1-4, 0..1 with 0.5 = as usual
  attribute vec4 a1;            // steps 5-8
  attribute vec4 a2;            // steps 9-12
  uniform float uStep;          // 0..11, fractional while replaying
  uniform float uSize;
  uniform float uSpan;          // a change this big counts as fully lit (this think's 95th percentile)
  varying vec3 vColor;
  varying float vAlpha;

  float at(float s) {
    vec4 i = vec4(0.0, 1.0, 2.0, 3.0);
    return dot(a0, vec4(equal(i, vec4(s)))) + dot(a1, vec4(equal(i + 4.0, vec4(s)))) + dot(a2, vec4(equal(i + 8.0, vec4(s))));
  }

  void main() {
    float s = floor(uStep);
    float d = mix(at(s), at(min(s + 1.0, 11.0)), uStep - s) * 2.0 - 1.0;   // -1 much less .. +1 much more
    float heat = smoothstep(0.3, 1.0, abs(d) / uSpan);
    vColor = mix(vec3(0.34, 0.36, 0.4), d > 0.0 ? vec3(0.96, 0.38, 0.33) : vec3(0.3, 0.6, 1.0), heat);
    vAlpha = 0.18 + 0.82 * heat;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = uSize * (0.6 + 1.0 * heat) / -mv.z;
  }`

const fragmentShader = /* glsl */ `
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    vec2 c = gl_PointCoord - 0.5;
    float r = dot(c, c);
    if (r > 0.25) discard;
    gl_FragColor = vec4(vColor, vAlpha * smoothstep(0.25, 0.0, r));
  }`

export class BrainView {
  constructor(canvas, meta, buffer) {
    this.canvas = canvas
    this.steps = meta.steps
    const n = (this.count = meta.count)
    const xyz = new Float32Array(buffer, 0, n * 3)
    const positions = new Float32Array(n * 3)
    for (let i = 0; i < n; i++) { // the data's z runs from the brain down the nerve cord: show it vertically
      positions.set([xyz[i * 3], -xyz[i * 3 + 2], xyz[i * 3 + 1]], i * 3)
    }
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    this.attrs = [0, 1, 2].map((k) => {
      const a = new THREE.BufferAttribute(new Uint8Array(n * 4).fill(128), 4, true)
      geometry.setAttribute(`a${k}`, a)
      return a
    })
    this.material = new THREE.ShaderMaterial({
      vertexShader, fragmentShader, transparent: true, depthWrite: false, // normal blending: 100k points can't pile up into white
      uniforms: { uStep: { value: 0 }, uSize: { value: 9 }, uSpan: { value: 1 } },
    })
    this.points = new THREE.Points(geometry, this.material)
    this.points.frustumCulled = false

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: false })
    this.renderer.setClearColor(0x1b1a18)
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2))
    this.scene = new THREE.Scene()
    this.camera = new THREE.PerspectiveCamera(30, 1, 0.1, 20)
    this.camera.position.set(0, 0.05, 4.1) // the brain and the whole nerve cord fit
    this.turntable = new THREE.Group()
    this.turntable.add(this.points)
    this.scene.add(this.turntable)
    this.spin = !matchMedia('(prefers-reduced-motion: reduce)').matches // a slow turn, unless you'd rather nothing moved
    this.dirty = true
    this.replay = null
    this.onStep = () => {}

    let dragX = null
    canvas.addEventListener('pointerdown', (e) => { dragX = e.clientX; canvas.setPointerCapture(e.pointerId) })
    canvas.addEventListener('pointermove', (e) => {
      if (dragX === null) return
      this.turntable.rotation.y += (e.clientX - dragX) * 0.01
      dragX = e.clientX
      this.dirty = true
    })
    canvas.addEventListener('pointerup', () => (dragX = null))
    new ResizeObserver(() => this.resize()).observe(canvas)
  }

  resize() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight
    if (!w || !h) return
    this.renderer.setSize(w, h, false)
    this.camera.aspect = w / h
    this.camera.updateProjectionMatrix()
    this.material.uniforms.uSize.value = 9 * h / 280 * this.renderer.getPixelRatio()
    this.dirty = true
  }

  /** bytes: steps x neurons, 128 = as usual. */
  setActivity(bytes) {
    const n = this.count
    const last = Array.from({ length: 4096 }, () => Math.abs(bytes[(this.steps - 1) * n + Math.floor(Math.random() * n)] / 127.5 - 1))
    this.material.uniforms.uSpan.value = Math.max(0.04, last.sort((a, b) => a - b)[Math.floor(last.length * 0.95)])
    for (let s = 0; s < this.steps; s++) { // step s goes to channel s % 4 of attribute s / 4
      const arr = this.attrs[s >> 2].array, row = bytes.subarray(s * n, (s + 1) * n)
      for (let i = 0, j = s & 3; i < n; i++, j += 4) arr[j] = row[i]
    }
    for (const a of this.attrs) a.needsUpdate = true
    this.dirty = true
  }

  setStep(s) {
    this.material.uniforms.uStep.value = s
    this.onStep(s)
    this.dirty = true
  }

  /** Replay the think from the eye (step 1) to the legs (step 12). */
  play(seconds = 1.4) {
    this.replay?.done() // a replay still running ends here; whoever waited for it sees the game moved on
    return new Promise((done) => (this.replay = { t: 0, seconds, done }))
  }

  render(dt) {
    if (this.replay) {
      this.replay.t += dt
      const k = Math.min(1, this.replay.t / this.replay.seconds)
      this.setStep(k * (this.steps - 1))
      if (k === 1) {
        this.replay.done()
        this.replay = null
      }
    }
    if (this.spin) {
      this.turntable.rotation.y += dt * 0.12
      this.dirty = true
    }
    if (!this.dirty || !this.canvas.clientWidth) return // hidden: the replay above still runs, so the fly still moves
    this.renderer.render(this.scene, this.camera)
    this.dirty = false
  }
}
