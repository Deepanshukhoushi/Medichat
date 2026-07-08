import { afterNextRender, Directive, ElementRef, inject, input, OnDestroy } from '@angular/core';

@Directive({
  selector: '[mcCountUp]',
  standalone: true
})
export class CountUpDirective implements OnDestroy {
  readonly mcCountUp = input.required<number>();
  readonly duration = input<number>(2000);
  readonly suffix = input<string>('');
  readonly label = input<string>('');
  
  private readonly elementRef = inject(ElementRef<HTMLElement>);
  private observer: IntersectionObserver | null = null;
  private hasAnimated = false;

  constructor() {
    afterNextRender(() => {
      const element = this.elementRef.nativeElement;
      
      if (typeof IntersectionObserver === 'undefined') {
        element.textContent = this.label() + this.suffix();
        return;
      }

      this.observer = new IntersectionObserver(
        (entries) => {
          for (const entry of entries) {
            if (entry.isIntersecting && !this.hasAnimated) {
              this.hasAnimated = true;
              this.animateCount();
              this.observer?.disconnect();
              this.observer = null;
            }
          }
        },
        { threshold: 0.1 }
      );

      this.observer.observe(element);
    });
  }

  private animateCount(): void {
    const element = this.elementRef.nativeElement;
    const target = this.mcCountUp();
    const duration = this.duration();
    const suffix = this.suffix();
    const label = this.label();
    
    // Fallback if not a number
    if (isNaN(target)) {
        element.textContent = label + suffix;
        return;
    }
    
    let startTimestamp: number | null = null;
    
    const step = (timestamp: number) => {
      if (!startTimestamp) startTimestamp = timestamp;
      const progress = Math.min((timestamp - startTimestamp) / duration, 1);
      
      // easeOutQuart
      const easeProgress = 1 - Math.pow(1 - progress, 4);
      const currentCount = Math.floor(easeProgress * target);
      
      let formattedCount = currentCount.toString();
      if (target >= 1000000) {
          formattedCount = (currentCount / 1000000).toFixed(1) + 'M';
      } else if (target >= 10000) {
          formattedCount = (currentCount / 1000).toFixed(0) + 'k';
      }
      
      element.textContent = formattedCount + suffix;
      
      if (progress < 1) {
        window.requestAnimationFrame(step);
      } else {
        element.textContent = label + suffix;
      }
    };
    
    window.requestAnimationFrame(step);
  }

  ngOnDestroy(): void {
    this.observer?.disconnect();
  }
}
