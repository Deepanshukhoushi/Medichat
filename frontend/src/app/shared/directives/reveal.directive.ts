import { afterNextRender, Directive, ElementRef, inject, input, OnDestroy } from '@angular/core';

@Directive({
  selector: '[mcReveal]',
  standalone: true
})
export class RevealDirective implements OnDestroy {
  readonly delay = input(0);
  private readonly elementRef = inject(ElementRef<HTMLElement>);
  private observer: IntersectionObserver | null = null;

  constructor() {
    const element = this.elementRef.nativeElement;
    element.style.opacity = '0';
    element.style.transform = 'translateY(24px)';
    element.style.transition = 'opacity 700ms ease, transform 700ms ease';

    afterNextRender(() => {
      if (typeof IntersectionObserver === 'undefined') {
        element.style.opacity = '1';
        element.style.transform = 'none';
        return;
      }

      this.observer = new IntersectionObserver(
        (entries) => {
          for (const entry of entries) {
            if (entry.isIntersecting) {
              window.setTimeout(() => {
                element.style.opacity = '1';
                element.style.transform = 'none';
              }, this.delay() * 1000);
              this.observer?.disconnect();
              this.observer = null;
            }
          }
        },
        { threshold: 0.15 }
      );

      this.observer.observe(element);
    });
  }

  ngOnDestroy(): void {
    this.observer?.disconnect();
  }
}
