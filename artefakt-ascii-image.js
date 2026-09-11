/* =====================================================================
 * ASCII FLOW FIELD — versão IMAGEM / LOGO
 * Recriação do efeito de hero do artefakt.mov, alimentado por um PNG/SVG
 * em vez de texto. Arquivo independente do artefakt-ascii.js — os dois
 * podem conviver na mesma página.
 *
 * Como funciona:
 *   1. A imagem é desenhada num canvas 2D invisível; cada pixel
 *      considerado "tinta" vira uma partícula (e guarda a cor de origem).
 *   2. As partículas são simuladas na GPU (ping-pong de render targets):
 *      um flow field de simplex noise 4D as empurra, elas morrem e
 *      renascem na posição original, e o mouse as repele.
 *   3. Um shader de pós-processamento troca cada célula do buffer por um
 *      caractere, escolhido pela luminância da célula.
 *
 * USO
 *   <div id="ascii-logo" data-src="https://.../logo.png"></div>
 *   <script type="module">import '.../artefakt-ascii-image.js';</script>
 *
 * A imagem PRECISA permitir CORS (o canvas é lido pixel a pixel). Assets
 * do Webflow servem com CORS liberado; um host próprio pode não servir.
 * Se o console acusar "tainted canvas", hospede a imagem no mesmo
 * domínio ou no Asset Manager do Webflow.
 * ===================================================================== */

import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.169.0/build/three.module.js';

/* ------------------------------------------------------------------ *
 *  CONFIG
 * ------------------------------------------------------------------ */
const CFG = {
  src:             '',          // URL da imagem (ou atributo data-src)
  fitWidth:        0.55,        // largura da imagem como fração da tela
  fitHeight:       0.80,        // teto de altura como fração da tela
  sampleStep:      2,           // menor = mais partículas = mais denso

  // como decidir o que é "tinta" na imagem:
  //   'auto'    – opaco e não-esbranquiçado  (logo sobre branco ou transparente)
  //   'alpha'   – qualquer pixel opaco       (PNG recortado, sem fundo branco)
  //   'dark'    – opaco e escuro             (arte preta sobre fundo claro)
  //   'light'   – opaco e claro              (arte branca sobre fundo escuro)
  //   'colored' – opaco e com saturação      (ignora branco E preto)
  pick:            'auto',
  threshold:       0.90,        // corte de luminância usado por auto/dark/light
  saturation:      0.08,        // corte de saturação usado por auto/colored

  // cor dos caracteres:
  //   colorMode 'source' – herda a cor do pixel de origem
  //   colorMode 'ink'    – cor única, definida em inkColor
  colorMode:       'ink',
  inkColor:        '#ffffff',
  theme:           'dark',      // 'dark' clareia os caracteres; 'light' não

  // quanto o claro/escuro da arte original vira densidade de caracteres.
  // 0 = silhueta chapada (tudo com a mesma densidade)
  // 1 = contraste total (áreas escuras da arte ficam bem mais rarefeitas)
  shading:         1.0,
  // inverte quem vira densidade. Num fundo escuro, a parte CLARA da arte é
  // que "existe" e a escura vira vazio. Se o desenho estiver na cor escura
  // (logo azul-marinho sobre fundo claro, p.ex.), ligue isto.
  shadingInvert:   false,

  intensity:       1.6,         // brilho por partícula (controla qual caractere sai)
  asciiChars:      ' .:-=+*#%@4RT3F',
  columns:         180,         // nº de colunas de caracteres (90 no mobile)
  renderScale:     0.3,         // escala do buffer interno (0.9 no mobile)
  contrast:        1.09,
  brightness:      0.0,
  flowInfluence:   0.43,
  flowStrength:    1.09,
  flowFrequency:   0.53,
  lifeSpeed:       0.32,
  mouseRadius:     0.22,        // fração da altura da tela
  mouseStrength:   2.6,
  pointSize:       1.5
};

/* ------------------------------------------------------------------ *
 *  GLSL: simplex noise 4D (Ashima / Stefan Gustavson — MIT)
 * ------------------------------------------------------------------ */
const NOISE = /* glsl */`
vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
float permute(float x){return floor(mod(((x*34.0)+1.0)*x, 289.0));}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}
float taylorInvSqrt(float r){return 1.79284291400159 - 0.85373472095314 * r;}
vec4 grad4(float j, vec4 ip){
  const vec4 ones = vec4(1.0, 1.0, 1.0, -1.0);
  vec4 p,s;
  p.xyz = floor( fract (vec3(j) * ip.xyz) * 7.0) * ip.z - 1.0;
  p.w = 1.5 - dot(abs(p.xyz), ones.xyz);
  s = vec4(lessThan(p, vec4(0.0)));
  p.xyz = p.xyz + (s.xyz*2.0 - 1.0) * s.www;
  return p;
}
float simplexNoise4d(vec4 v){
  const vec2 C = vec2(0.138196601125010504, 0.309016994374947451);
  vec4 i  = floor(v + dot(v, C.yyyy));
  vec4 x0 = v -   i + dot(i, C.xxxx);
  vec4 i0;
  vec3 isX  = step(x0.yzw, x0.xxx);
  vec3 isYZ = step(x0.zww, x0.yyz);
  i0.x = isX.x + isX.y + isX.z;
  i0.yzw = 1.0 - isX;
  i0.y += isYZ.x + isYZ.y;
  i0.zw += 1.0 - isYZ.xy;
  i0.z += isYZ.z;
  i0.w += 1.0 - isYZ.z;
  vec4 i3 = clamp(i0, 0.0, 1.0);
  vec4 i2 = clamp(i0-1.0, 0.0, 1.0);
  vec4 i1 = clamp(i0-2.0, 0.0, 1.0);
  vec4 x1 = x0 - i1 + 1.0 * C.xxxx;
  vec4 x2 = x0 - i2 + 2.0 * C.xxxx;
  vec4 x3 = x0 - i3 + 3.0 * C.xxxx;
  vec4 x4 = x0 - 1.0 + 4.0 * C.xxxx;
  i = mod(i, 289.0);
  float j0 = permute(permute(permute(permute(i.w) + i.z) + i.y) + i.x);
  vec4 j1 = permute(permute(permute(permute(
             i.w + vec4(i1.w, i2.w, i3.w, 1.0))
           + i.z + vec4(i1.z, i2.z, i3.z, 1.0))
           + i.y + vec4(i1.y, i2.y, i3.y, 1.0))
           + i.x + vec4(i1.x, i2.x, i3.x, 1.0));
  vec4 ip = vec4(1.0/294.0, 1.0/49.0, 1.0/7.0, 0.0);
  vec4 p0 = grad4(j0,   ip);
  vec4 p1 = grad4(j1.x, ip);
  vec4 p2 = grad4(j1.y, ip);
  vec4 p3 = grad4(j1.z, ip);
  vec4 p4 = grad4(j1.w, ip);
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
  p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
  p4 *= taylorInvSqrt(dot(p4,p4));
  vec3 m0 = max(0.6 - vec3(dot(x0,x0), dot(x1,x1), dot(x2,x2)), 0.0);
  vec2 m1 = max(0.6 - vec2(dot(x3,x3), dot(x4,x4)), 0.0);
  m0 = m0 * m0; m1 = m1 * m1;
  return 49.0 * (dot(m0*m0, vec3(dot(p0,x0), dot(p1,x1), dot(p2,x2)))
               + dot(m1*m1, vec2(dot(p3,x3), dot(p4,x4))));
}`;

/* ------------------------------------------------------------------ *
 *  BOOT
 * ------------------------------------------------------------------ */
const root = document.getElementById('ascii-logo');
if (!root) throw new Error('[ascii] elemento #ascii-logo não encontrado');

if (root.dataset.src)   CFG.src = root.dataset.src;
if (root.dataset.pick)  CFG.pick = root.dataset.pick;
if (root.dataset.color) CFG.colorMode = root.dataset.color;
if (!CFG.src) throw new Error('[ascii] defina data-src no #ascii-logo');

const isMobile = window.matchMedia('(max-width: 767px)').matches;
if (isMobile) { CFG.columns = 90; CFG.renderScale = 0.9; CFG.fitWidth = 0.85; }

const image = new Image();
image.crossOrigin = 'anonymous';
image.onerror = () => console.error('[ascii] falha ao carregar a imagem:', CFG.src);
image.onload = () => { try { init(image); } catch (e) { console.error('[ascii]', e); } };
image.src = CFG.src;

/* ------------------------------------------------------------------ *
 *  1. AMOSTRAGEM DA IMAGEM  →  nuvem de pontos + cores
 * ------------------------------------------------------------------ */
function hexToRgb(hex) {
  const h = hex.replace('#', '');
  const n = parseInt(h.length === 3 ? h.split('').map(c => c + c).join('') : h, 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

function sampleImage(img, W, H) {
  // tamanho de desenho: respeita a proporção original
  let dw = W * CFG.fitWidth;
  let dh = dw * (img.naturalHeight / img.naturalWidth);
  const maxH = H * CFG.fitHeight;
  if (dh > maxH) { dh = maxH; dw = dh * (img.naturalWidth / img.naturalHeight); }
  dw = Math.round(dw); dh = Math.round(dh);

  const c = document.createElement('canvas');
  c.width = dw; c.height = dh;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, dw, dh);

  let data;
  try {
    data = ctx.getImageData(0, 0, dw, dh).data;
  } catch (e) {
    throw new Error('canvas bloqueado por CORS — hospede a imagem no mesmo domínio ' +
                    'ou num host que envie Access-Control-Allow-Origin');
  }

  const ink = hexToRgb(CFG.inkColor);
  const useSource = CFG.colorMode === 'source';
  const pts = [], cols = [], lums = [];
  const s = CFG.sampleStep;
  let maxLuma = 0.001;

  for (let y = 0; y < dh; y += s) {
    for (let x = 0; x < dw; x += s) {
      const i = (y * dw + x) * 4;
      const a = data[i + 3] / 255;
      if (a < 0.5) continue;

      const r = data[i] / 255, g = data[i + 1] / 255, b = data[i + 2] / 255;
      const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      const sat  = Math.max(r, g, b) - Math.min(r, g, b);

      let keep;
      switch (CFG.pick) {
        case 'alpha':   keep = true; break;
        case 'dark':    keep = luma < CFG.threshold; break;
        case 'light':   keep = luma > (1 - CFG.threshold); break;
        case 'colored': keep = sat > CFG.saturation; break;
        default:        keep = luma < CFG.threshold || sat > CFG.saturation; // auto
      }
      if (!keep) continue;

      pts.push(x - dw / 2, -(y - dh / 2), (Math.random() - 0.5) * 30);
      lums.push(luma);
      if (luma > maxLuma) maxLuma = luma;

      if (useSource) {
        // preserva o matiz e joga o brilho para o topo: um azul-marinho
        // escuro precisa render caractere legível sobre fundo preto.
        // O claro/escuro original volta depois, como densidade (shading).
        const m = Math.max(r, g, b, 0.001);
        cols.push(r / m, g / m, b / m);
      } else {
        cols.push(ink[0], ink[1], ink[2]);
      }
    }
  }

  // normaliza pela cor mais clara da arte: um logo de cor única fica
  // uniforme, e um logo de duas cores mantém a hierarquia entre elas
  const sh = CFG.shading;
  for (let i = 0; i < lums.length; i++) {
    let v = lums[i] / maxLuma;
    if (CFG.shadingInvert) v = 1 - v * 0.92;   // 0.92 evita zerar a cor mais clara
    lums[i] = (1 - sh) + sh * v;
  }

  return { pts, cols, lums, dw, dh };
}

/* ------------------------------------------------------------------ *
 *  2. CENA
 * ------------------------------------------------------------------ */
function init(img) {
  let W = root.clientWidth, H = root.clientHeight;

  const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(W, H);
  renderer.setClearColor(0x000000, 0);
  root.appendChild(renderer.domElement);

  const scene  = new THREE.Scene();
  const camera = new THREE.OrthographicCamera(-W/2, W/2, H/2, -H/2, -5000, 5000);

  const { pts, cols, lums } = sampleImage(img, W, H);
  const count = pts.length / 3;
  if (!count) { console.warn('[ascii] nenhum pixel passou no filtro "' + CFG.pick + '"'); return; }
  const fboSize = Math.ceil(Math.sqrt(count));

  /* --- textura-base com as posições de origem --- */
  const baseData = new Float32Array(fboSize * fboSize * 4);
  for (let i = 0; i < fboSize * fboSize; i++) {
    if (i < count) {
      baseData[i*4+0] = pts[i*3+0];
      baseData[i*4+1] = pts[i*3+1];
      baseData[i*4+2] = pts[i*3+2];
      baseData[i*4+3] = Math.random();          // offset de "vida"
    } else {
      baseData[i*4+3] = -1.0;                   // slot vazio
    }
  }
  const baseTexture = new THREE.DataTexture(baseData, fboSize, fboSize, THREE.RGBAFormat, THREE.FloatType);
  baseTexture.needsUpdate = true;

  /* --- ping-pong FBO: simulação --- */
  const rtOpts = {
    minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter,
    format: THREE.RGBAFormat, type: THREE.FloatType, depthBuffer: false
  };
  let rtA = new THREE.WebGLRenderTarget(fboSize, fboSize, rtOpts);
  let rtB = new THREE.WebGLRenderTarget(fboSize, fboSize, rtOpts);

  const simScene  = new THREE.Scene();
  const simCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

  const simMaterial = new THREE.ShaderMaterial({
    uniforms: {
      uParticles:     { value: null },
      uBase:          { value: baseTexture },
      uTime:          { value: 0 },
      uDelta:         { value: 0 },
      uFlowInfluence: { value: CFG.flowInfluence },
      uFlowStrength:  { value: CFG.flowStrength },
      uFlowFrequency: { value: CFG.flowFrequency },
      uLifeSpeed:     { value: CFG.lifeSpeed },
      uMouse:         { value: new THREE.Vector3(0, 0, 0) },
      uMouseRadius:   { value: H * CFG.mouseRadius },
      uMouseStrength: { value: CFG.mouseStrength },
      uScale:         { value: 1 }
    },
    vertexShader: `
      varying vec2 vUv;
      void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }
    `,
    fragmentShader: NOISE + `
      uniform sampler2D uParticles;
      uniform sampler2D uBase;
      uniform float uTime, uDelta, uLifeSpeed;
      uniform float uFlowInfluence, uFlowStrength, uFlowFrequency;
      uniform vec3  uMouse;
      uniform float uMouseRadius, uMouseStrength, uScale;
      varying vec2 vUv;

      void main(){
        vec4 particle = texture2D(uParticles, vUv);
        vec4 base     = texture2D(uBase, vUv);

        if (base.a < 0.0) { gl_FragColor = vec4(0.0, 0.0, 0.0, -1.0); return; }

        if (particle.a >= 1.0) {
          particle.a   = fract(particle.a);
          particle.xyz = base.xyz;
        } else {
          float freq = uFlowFrequency * 0.006 / uScale;

          float strength = simplexNoise4d(vec4(base.xyz * freq * 0.4, uTime + 1.0));
          float influence = (uFlowInfluence - 0.5) * (-2.0);
          strength = smoothstep(influence, 1.0, strength);

          vec3 flow = vec3(
            simplexNoise4d(vec4(particle.xyz * freq + 0.0, uTime * 0.25)),
            simplexNoise4d(vec4(particle.xyz * freq + 1.0, uTime * 0.25)),
            simplexNoise4d(vec4(particle.xyz * freq + 2.0, uTime * 0.25))
          );
          particle.xyz += normalize(flow) * uDelta * strength * uFlowStrength * 32.0 * uScale;

          vec2  d    = particle.xy - uMouse.xy;
          float dist = length(d);
          float f    = 1.0 - smoothstep(0.0, uMouseRadius, dist);
          particle.xy += normalize(d + 0.0001) * f * f * uMouseStrength * uMouseRadius * uDelta;

          particle.a += uDelta * uLifeSpeed;
        }
        gl_FragColor = particle;
      }
    `
  });
  simScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), simMaterial));

  (function seed(){
    const init = new THREE.DataTexture(baseData.slice(), fboSize, fboSize, THREE.RGBAFormat, THREE.FloatType);
    init.needsUpdate = true;
    simMaterial.uniforms.uParticles.value = init;
    simMaterial.uniforms.uDelta.value = 0;
    renderer.setRenderTarget(rtA); renderer.render(simScene, simCamera);
    renderer.setRenderTarget(rtB); renderer.render(simScene, simCamera);
    renderer.setRenderTarget(null);
  })();

  /* --- partículas --- */
  const geo = new THREE.BufferGeometry();
  const refs  = new Float32Array(count * 2);
  const seeds = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    refs[i*2+0] = (i % fboSize) / fboSize + 0.5 / fboSize;
    refs[i*2+1] = Math.floor(i / fboSize) / fboSize + 0.5 / fboSize;
    seeds[i] = (0.55 + Math.random() * 0.45) * lums[i];
  }
  geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(count * 3), 3));
  geo.setAttribute('aRef',   new THREE.BufferAttribute(refs, 2));
  geo.setAttribute('aSeed',  new THREE.BufferAttribute(seeds, 1));
  geo.setAttribute('aColor', new THREE.BufferAttribute(new Float32Array(cols), 3));

  const particles = new THREE.Points(geo, new THREE.ShaderMaterial({
    uniforms: {
      uParticles: { value: null },
      uPointSize: { value: CFG.pointSize },
      uIntensity: { value: CFG.intensity },
      uTime:      { value: 0 },
      uScale:     { value: renderer.getPixelRatio() * CFG.renderScale }
    },
    vertexShader: NOISE + `
      uniform sampler2D uParticles;
      uniform float uPointSize, uScale, uIntensity, uTime;
      attribute vec2 aRef;
      attribute float aSeed;
      attribute vec3 aColor;
      varying float vAlpha;
      varying vec3  vColor;
      void main(){
        vec4 p = texture2D(uParticles, aRef);
        if (p.a < 0.0) { gl_Position = vec4(2.0); vAlpha = 0.0; return; }
        float fadeIn  = smoothstep(0.0, 0.12, p.a);
        float fadeOut = 1.0 - smoothstep(0.85, 1.0, p.a);
        float shade = simplexNoise4d(vec4(p.xyz * 0.006, uTime * 0.12)) * 0.5 + 0.5;
        vAlpha = fadeIn * fadeOut * aSeed * (0.22 + shade * 1.05) * uIntensity;
        vColor = aColor;
        vec4 mv = modelViewMatrix * vec4(p.xyz, 1.0);
        gl_Position = projectionMatrix * mv;
        gl_PointSize = uPointSize * uScale;
      }
    `,
    fragmentShader: `
      varying float vAlpha;
      varying vec3  vColor;
      void main(){ gl_FragColor = vec4(vColor * vAlpha, vAlpha); }
    `,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending
  }));
  particles.frustumCulled = false;
  scene.add(particles);

  /* --- atlas de caracteres --- */
  function makeAsciiTexture(chars) {
    const c = document.createElement('canvas');
    c.width = 16 * chars.length; c.height = 16;
    const ctx = c.getContext('2d');
    ctx.fillStyle = 'black'; ctx.fillRect(0, 0, c.width, c.height);
    ctx.fillStyle = 'white'; ctx.font = '16px monospace';
    ctx.textBaseline = 'middle'; ctx.textAlign = 'center';
    chars.split('').forEach((ch, i) => ctx.fillText(ch, 16 * (i + 0.5), 8));
    const tex = new THREE.CanvasTexture(c);
    tex.minFilter = THREE.NearestFilter;
    tex.magFilter = THREE.NearestFilter;
    return tex;
  }

  /* --- passe ASCII --- */
  let bufW = Math.round(W * CFG.renderScale);
  let bufH = Math.round(H * CFG.renderScale);
  const sceneRT = new THREE.WebGLRenderTarget(bufW, bufH, {
    minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter
  });

  const asciiMaterial = new THREE.ShaderMaterial({
    transparent: true,
    uniforms: {
      tDiffuse:         { value: sceneRT.texture },
      uResolution:      { value: new THREE.Vector2(bufW, bufH) },
      uAsciiTexture:    { value: makeAsciiTexture(CFG.asciiChars) },
      uCharCount:       { value: new THREE.Vector2(CFG.asciiChars.length, 1) },
      uAsciiPixelSize:  { value: bufW / CFG.columns },
      uAsciiBrightness: { value: CFG.brightness },
      uAsciiContrast:   { value: CFG.contrast },
      uAsciiMin:        { value: 0 },
      uAsciiMax:        { value: 1 },
      uGain:            { value: CFG.theme === 'dark' ? 1 : 0 }
    },
    vertexShader: `
      varying vec2 vUv;
      void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }
    `,
    fragmentShader: `
      uniform sampler2D tDiffuse;
      uniform vec2  uResolution;
      uniform float uAsciiPixelSize;
      uniform sampler2D uAsciiTexture;
      uniform vec2  uCharCount;
      uniform float uAsciiContrast, uAsciiBrightness, uAsciiMin, uAsciiMax, uGain;
      varying vec2 vUv;

      void main() {
        vec2 normalizedPixelSize = uAsciiPixelSize / uResolution;
        vec2 uvPixel = normalizedPixelSize * floor(vUv / normalizedPixelSize);
        vec4 texColor = texture2D(tDiffuse, uvPixel);

        float luma = dot(vec3(0.2126, 0.7152, 0.0722), texColor.rgb);

        luma = (luma - uAsciiMin) / (uAsciiMax - uAsciiMin);
        luma = clamp(luma, 0.0, 1.0);
        luma = luma + uAsciiBrightness;
        luma = (luma - 0.5) * uAsciiContrast + 0.5;
        luma = clamp(luma, 0.0, 1.0);

        vec2 cellUV = fract(vUv / normalizedPixelSize);

        float charIndex = clamp(
          floor(luma * (uCharCount.x - 1.0)),
          0.0,
          uCharCount.x - 1.0
        );

        vec2 asciiUV = vec2((charIndex + cellUV.x) / uCharCount.x, cellUV.y);
        float character = texture2D(uAsciiTexture, asciiUV).r;

        // recupera o matiz da célula sem o escurecimento da densidade
        float peak = max(texColor.r, max(texColor.g, texColor.b));
        vec3  hue  = texColor.rgb / max(peak, 0.001);

        float gain = mix(1.0, luma + 0.9, uGain);
        vec3  finalColor = character * hue * gain;
        float alpha = texColor.a * character;

        gl_FragColor = vec4(finalColor, alpha);
      }
    `
  });
  const asciiScene = new THREE.Scene();
  asciiScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), asciiMaterial));

  /* --- mouse --- */
  const mouse = new THREE.Vector2(-9999, -9999);
  const mouseLerp = new THREE.Vector2(-9999, -9999);
  window.addEventListener('pointermove', (e) => {
    const r = root.getBoundingClientRect();
    mouse.set(e.clientX - r.left - W/2, -(e.clientY - r.top - H/2));
  }, { passive: true });
  root.addEventListener('pointerleave', () => mouse.set(-9999, -9999));

  /* --- loop --- */
  const clock = new THREE.Clock();
  let prev = 0;
  function tick() {
    const t  = clock.getElapsedTime();
    const dt = Math.min(t - prev, 1/30);
    prev = t;

    mouseLerp.lerp(mouse, 0.12);
    simMaterial.uniforms.uMouse.value.set(mouseLerp.x, mouseLerp.y, 0);
    simMaterial.uniforms.uTime.value  = t;
    simMaterial.uniforms.uDelta.value = dt;
    simMaterial.uniforms.uParticles.value = rtA.texture;

    renderer.setRenderTarget(rtB);
    renderer.render(simScene, simCamera);
    renderer.setRenderTarget(null);
    const tmp = rtA; rtA = rtB; rtB = tmp;

    particles.material.uniforms.uParticles.value = rtA.texture;
    particles.material.uniforms.uTime.value = t;

    renderer.setRenderTarget(sceneRT);
    renderer.clear();
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);

    renderer.render(asciiScene, simCamera);
    requestAnimationFrame(tick);
  }
  tick();

  /* --- resize --- */
  let rzT;
  window.addEventListener('resize', () => {
    clearTimeout(rzT);
    rzT = setTimeout(() => {
      W = root.clientWidth; H = root.clientHeight;
      renderer.setSize(W, H);
      camera.left = -W/2; camera.right = W/2; camera.top = H/2; camera.bottom = -H/2;
      camera.updateProjectionMatrix();
      bufW = Math.round(W * CFG.renderScale);
      bufH = Math.round(H * CFG.renderScale);
      sceneRT.setSize(bufW, bufH);
      asciiMaterial.uniforms.uResolution.value.set(bufW, bufH);
      asciiMaterial.uniforms.uAsciiPixelSize.value = bufW / CFG.columns;
      simMaterial.uniforms.uMouseRadius.value = H * CFG.mouseRadius;
    }, 200);
  });
}
