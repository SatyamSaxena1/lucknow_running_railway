import * as THREE from "three";

const filters = document.querySelectorAll(".filter");
const cards = document.querySelectorAll(".project-card");

filters.forEach((filter) => {
  filter.addEventListener("click", () => {
    const selected = filter.dataset.filter;

    filters.forEach((item) => {
      const isActive = item === filter;
      item.classList.toggle("active", isActive);
      item.setAttribute("aria-selected", String(isActive));
    });

    cards.forEach((card) => {
      const tags = card.dataset.tags.split(" ");
      const shouldShow = selected === "all" || tags.includes(selected);
      card.hidden = !shouldShow;
    });
  });
});

const canvas = document.querySelector("#rail-scene");

if (canvas) {
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: true,
    preserveDrawingBuffer: true,
    powerPreference: "high-performance",
  });

  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x000000, 0);

  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x152632, 18, 42);

  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
  camera.position.set(0, 5.2, 16);
  camera.lookAt(0, 2.2, 0);

  const root = new THREE.Group();
  scene.add(root);

  const ambient = new THREE.HemisphereLight(0xf3f8ff, 0x1b2a22, 2.8);
  scene.add(ambient);

  const key = new THREE.DirectionalLight(0xffffff, 3.2);
  key.position.set(-5, 7, 8);
  scene.add(key);

  const railMaterial = new THREE.MeshStandardMaterial({
    color: 0xc7d0cf,
    metalness: 0.72,
    roughness: 0.28,
  });
  const sleeperMaterial = new THREE.MeshStandardMaterial({
    color: 0x6c5f52,
    metalness: 0.08,
    roughness: 0.7,
  });
  const trainMaterial = new THREE.MeshStandardMaterial({
    color: 0x273846,
    metalness: 0.4,
    roughness: 0.42,
  });
  const redMaterial = new THREE.MeshStandardMaterial({
    color: 0xbe3438,
    emissive: 0x3a0708,
    metalness: 0.32,
    roughness: 0.36,
  });
  const wireMaterial = new THREE.MeshStandardMaterial({
    color: 0xf6d58b,
    emissive: 0x9e6217,
    emissiveIntensity: 0.6,
    metalness: 0.28,
    roughness: 0.25,
  });
  const scanMaterial = new THREE.MeshBasicMaterial({
    color: 0x58d9cf,
    transparent: true,
    opacity: 0.36,
    side: THREE.DoubleSide,
    depthWrite: false,
  });

  const makeBox = (size, position, material) => {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(...size), material);
    mesh.position.set(...position);
    root.add(mesh);
    return mesh;
  };

  makeBox([0.16, 0.16, 34], [-2.1, 0.12, 0], railMaterial);
  makeBox([0.16, 0.16, 34], [2.1, 0.12, 0], railMaterial);

  for (let z = -14; z <= 14; z += 1.8) {
    makeBox([5.6, 0.12, 0.28], [0, -0.05, z], sleeperMaterial);
  }

  const roof = makeBox([4.6, 0.34, 7.8], [0, 0.68, 2.4], trainMaterial);
  roof.rotation.x = -0.02;
  makeBox([3.8, 0.24, 1.4], [0, 1.04, 0.4], redMaterial);

  const pantograph = new THREE.Group();
  pantograph.position.set(0, 1.2, 0.4);
  root.add(pantograph);

  const barGeometry = new THREE.CylinderGeometry(0.04, 0.04, 2.7, 16);
  const makeBar = (x, y, rotationZ) => {
    const bar = new THREE.Mesh(barGeometry, redMaterial);
    bar.position.set(x, y, 0);
    bar.rotation.z = rotationZ;
    pantograph.add(bar);
    return bar;
  };

  makeBar(-0.58, 1.05, -0.55);
  makeBar(0.58, 1.05, 0.55);
  makeBar(-0.58, 2.05, 0.55);
  makeBar(0.58, 2.05, -0.55);
  const contactStrip = new THREE.Mesh(new THREE.BoxGeometry(2.9, 0.09, 0.24), wireMaterial);
  contactStrip.position.set(0, 3.08, 0);
  pantograph.add(contactStrip);

  const wire = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 18, 16), wireMaterial);
  wire.position.set(0, 4.48, 0);
  wire.rotation.x = Math.PI / 2;
  root.add(wire);

  const mastMaterial = new THREE.MeshStandardMaterial({
    color: 0x8ea3a8,
    metalness: 0.55,
    roughness: 0.35,
  });

  for (const x of [-3.3, 3.3]) {
    const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.055, 5.1, 14), mastMaterial);
    mast.position.set(x, 2.4, -2.8);
    root.add(mast);
    makeBox([0.16, 0.12, 4.2], [x * 0.48, 4.56, -2.8], mastMaterial);
  }

  const scanPlane = new THREE.Mesh(new THREE.RingGeometry(1.1, 1.18, 56), scanMaterial);
  scanPlane.position.set(0, 3.12, 0);
  scanPlane.rotation.x = Math.PI / 2;
  pantograph.add(scanPlane);

  const scanCone = new THREE.Mesh(new THREE.ConeGeometry(2.5, 3.8, 48, 1, true), scanMaterial);
  scanCone.position.set(0, 2.38, 1.1);
  scanCone.rotation.x = Math.PI;
  scanCone.scale.z = 0.42;
  root.add(scanCone);

  const detectionBox = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(3.4, 2.1, 1.4)),
    new THREE.LineBasicMaterial({ color: 0x6ff7e9, transparent: true, opacity: 0.72 })
  );
  detectionBox.position.set(0, 2.48, 0.35);
  root.add(detectionBox);

  const gpsDots = new THREE.Group();
  for (let index = 0; index < 12; index += 1) {
    const dot = new THREE.Mesh(
      new THREE.SphereGeometry(0.055, 12, 12),
      new THREE.MeshBasicMaterial({ color: index % 3 === 0 ? 0xd38a2a : 0x58d9cf })
    );
    dot.position.set(Math.sin(index * 0.65) * 2.3, 0.32, -8 + index * 1.18);
    gpsDots.add(dot);
  }
  root.add(gpsDots);

  const pointer = { x: 0, y: 0 };
  const target = { x: 0, y: 0 };

  const resize = () => {
    const width = canvas.clientWidth || 640;
    const height = canvas.clientHeight || 420;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
  };

  const onPointerMove = (event) => {
    const rect = canvas.getBoundingClientRect();
    target.x = ((event.clientX - rect.left) / rect.width - 0.5) * 2;
    target.y = ((event.clientY - rect.top) / rect.height - 0.5) * 2;
  };

  window.addEventListener("resize", resize);
  canvas.addEventListener("pointermove", onPointerMove);
  resize();

  const clock = new THREE.Clock();

  const animate = () => {
    const elapsed = clock.getElapsedTime();
    pointer.x += (target.x - pointer.x) * 0.04;
    pointer.y += (target.y - pointer.y) * 0.04;

    root.rotation.y = pointer.x * 0.12 + Math.sin(elapsed * 0.28) * 0.035;
    root.rotation.x = -0.08 - pointer.y * 0.035;
    pantograph.position.y = 1.2 + Math.sin(elapsed * 1.6) * 0.045;
    scanPlane.rotation.z = elapsed * 0.9;
    scanPlane.scale.setScalar(1 + Math.sin(elapsed * 2.2) * 0.055);
    scanCone.material.opacity = 0.23 + Math.sin(elapsed * 2.4) * 0.08;
    detectionBox.rotation.y = Math.sin(elapsed * 0.9) * 0.05;
    gpsDots.children.forEach((dot, index) => {
      dot.position.y = 0.32 + Math.sin(elapsed * 2.1 + index * 0.7) * 0.06;
    });

    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  };

  animate();
}
