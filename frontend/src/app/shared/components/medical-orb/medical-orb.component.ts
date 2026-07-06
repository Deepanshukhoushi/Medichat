import { isPlatformBrowser, NgClass } from '@angular/common';
import {
  AfterViewInit,
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  ElementRef,
  PLATFORM_ID,
  ViewChild,
  inject,
  input
} from '@angular/core';
import * as THREE from 'three';

@Component({
  selector: 'mc-medical-orb',
  standalone: true,
  imports: [NgClass],
  templateUrl: './medical-orb.component.html',
  styleUrl: './medical-orb.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class MedicalOrbComponent implements AfterViewInit {
  readonly size = input<'hero' | 'compact'>('hero');
  readonly extraClass = input('');

  @ViewChild('orbCanvas', { static: true }) private readonly orbCanvas?: ElementRef<HTMLCanvasElement>;

  private readonly platformId = inject(PLATFORM_ID);
  private readonly destroyRef = inject(DestroyRef);

  ngAfterViewInit(): void {
    if (!isPlatformBrowser(this.platformId) || !this.orbCanvas) {
      return;
    }

    const canvas = this.orbCanvas.nativeElement;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    const geometry = new THREE.IcosahedronGeometry(1.4, 24);
    const material = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color('#f8b7d8'),
      transmission: 1,
      thickness: 1.2,
      roughness: 0.05,
      metalness: 0.06,
      clearcoat: 1,
      clearcoatRoughness: 0.08,
      iridescence: 1,
      iridescenceIOR: 1.3
    });
    const orb = new THREE.Mesh(geometry, material);
    scene.add(orb);

    const particleGeometry = new THREE.BufferGeometry();
    const particleCount = 220;
    const particlePositions = new Float32Array(particleCount * 3);
    for (let index = 0; index < particlePositions.length; index += 3) {
      particlePositions[index] = (Math.random() - 0.5) * 7;
      particlePositions[index + 1] = (Math.random() - 0.5) * 7;
      particlePositions[index + 2] = (Math.random() - 0.5) * 7;
    }

    particleGeometry.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));
    const particleMaterial = new THREE.PointsMaterial({
      color: '#7ee7ff',
      size: 0.03,
      transparent: true,
      opacity: 0.75
    });
    const particles = new THREE.Points(particleGeometry, particleMaterial);
    scene.add(particles);

    const ambient = new THREE.AmbientLight('#ffffff', 0.7);
    const pointLight = new THREE.PointLight('#58c7ff', 4, 100);
    pointLight.position.set(4, 3, 6);
    const accentLight = new THREE.PointLight('#ff7eb4', 3, 100);
    accentLight.position.set(-4, -3, 6);
    scene.add(ambient, pointLight, accentLight);

    camera.position.z = 5.2;

    const resize = () => {
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };

    resize();
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(canvas);

    let frameId = 0;
    const animate = () => {
      frameId = requestAnimationFrame(animate);
      const time = performance.now() * 0.001;
      orb.rotation.y = time * 0.38;
      orb.rotation.x = Math.sin(time * 0.4) * 0.18;
      orb.position.y = Math.sin(time) * 0.12;
      particles.rotation.y = time * 0.06;
      renderer.render(scene, camera);
    };

    animate();

    const handlePointerMove = (event: PointerEvent) => {
      const rect = canvas.getBoundingClientRect();
      const x = ((event.clientX - rect.left) / rect.width - 0.5) * 0.8;
      const y = ((event.clientY - rect.top) / rect.height - 0.5) * 0.8;
      orb.rotation.x = y * 0.55;
      orb.rotation.z = x * 0.25;
    };

    canvas.addEventListener('pointermove', handlePointerMove);

    this.destroyRef.onDestroy(() => {
      cancelAnimationFrame(frameId);
      resizeObserver.disconnect();
      canvas.removeEventListener('pointermove', handlePointerMove);
      geometry.dispose();
      material.dispose();
      particleGeometry.dispose();
      particleMaterial.dispose();
      renderer.dispose();
    });
  }
}
