import { Injectable, isDevMode } from '@angular/core';

/**
 * AnalyticsService
 *
 * Lightweight, consent-respecting Google Analytics 4 integration.
 *
 * Configuration:
 * ─────────────
 * Set the `medichat-ga-id` <meta> tag value in `index.html` to your GA4
 * Measurement ID (e.g. G-XXXXXXXXXX).
 *
 * In Vercel, set the environment variable:
 *   NG_APP_GA_ID=G-XXXXXXXXXX
 * and update the Angular build to inject it as the meta content value.
 *
 * Leave the meta content empty ("") to disable analytics entirely.
 *
 * Consent:
 * ────────
 * Analytics are only initialised after the user accepts cookies via the
 * CookieConsentComponent. The banner stores "accepted" / "declined" in
 * localStorage under the key "medichat-cookie-consent".
 *
 * Usage:
 * ──────
 *   // In AppComponent constructor or after consent is granted:
 *   inject(AnalyticsService).init();
 */
@Injectable({ providedIn: 'root' })
export class AnalyticsService {
  private static readonly CONSENT_KEY = 'medichat-cookie-consent';
  private initialized = false;

  /** Read GA Measurement ID from the meta tag injected into index.html. */
  private getMeasurementId(): string | null {
    if (typeof document === 'undefined') return null;
    const meta = document.querySelector<HTMLMetaElement>('meta[name="medichat-ga-id"]');
    return meta?.content?.trim() || null;
  }

  /**
   * Initialise GA4 only if:
   *  1. A Measurement ID is configured.
   *  2. The user has accepted cookies.
   *  3. Not running in dev mode (avoids polluting analytics with dev traffic).
   *  4. Not already initialised.
   */
  init(): void {
    if (this.initialized) return;
    if (isDevMode()) return;

    const measurementId = this.getMeasurementId();
    if (!measurementId) return;

    const consent = localStorage.getItem(AnalyticsService.CONSENT_KEY);
    if (consent !== 'accepted') return;

    this.loadGtag(measurementId);
    this.initialized = true;
  }

  /** Track a page view manually (useful for SPA navigation). */
  trackPageView(url: string): void {
    if (!this.initialized) return;
    this.gtag('event', 'page_view', { page_path: url });
  }

  /** Track a custom event. */
  trackEvent(eventName: string, params?: Record<string, unknown>): void {
    if (!this.initialized) return;
    this.gtag('event', eventName, params);
  }

  private loadGtag(measurementId: string): void {
    // Inject the gtag.js script
    const script = document.createElement('script');
    script.async = true;
    script.src = `https://www.googletagmanager.com/gtag/js?id=${measurementId}`;
    document.head.appendChild(script);

    // Initialise the dataLayer and gtag function
    (window as any).dataLayer = (window as any).dataLayer || [];
    (window as any).gtag = function () {
      // eslint-disable-next-line prefer-rest-params
      (window as any).dataLayer.push(arguments);
    };

    this.gtag('js', new Date());
    this.gtag('config', measurementId, {
      // Anonymize IP for GDPR compliance
      anonymize_ip: true,
      // Disable sending hits until after consent is confirmed
      send_page_view: false
    });
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private gtag(...args: any[]): void {
    if (typeof (window as any).gtag === 'function') {
      (window as any).gtag(...args);
    }
  }
}
