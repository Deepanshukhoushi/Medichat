import { Injectable, inject, signal } from '@angular/core';
import { Observable, shareReplay, tap } from 'rxjs';

import { BackendApiService, UserProfile } from './backend-api.service';

/**
 * Singleton cache for the authenticated user's profile.
 *
 * Problem solved: getProfile() was being called independently in 7 places
 * (guards, layouts, page components) causing 3+ redundant GET /api/profile/me
 * requests on every navigation to /app/chat.
 *
 * This service fetches the profile once (shareReplay(1)) and exposes it as:
 *   - `profile` signal  – for reactive template bindings
 *   - `profile$`        – for guards / RxJS pipelines
 *
 * Call `invalidate()` after login, logout, or profile updates so the next
 * subscriber triggers a fresh fetch.
 */
@Injectable({ providedIn: 'root' })
export class ProfileService {
  private readonly backendApi = inject(BackendApiService);

  /** Reactive signal — null until first fetch completes or on unauthenticated state. */
  readonly profile = signal<UserProfile | null>(null);

  private _profile$: Observable<UserProfile> | null = null;

  /**
   * Returns the shared, cached Observable. The first subscriber triggers the
   * HTTP call; all subsequent subscribers get the replayed value instantly.
   */
  get profile$(): Observable<UserProfile> {
    if (!this._profile$) {
      this._profile$ = this.backendApi.getProfile().pipe(
        tap((p) => this.profile.set(p)),
        shareReplay(1)
      );
    }
    return this._profile$;
  }

  /**
   * Clears the in-memory cache and resets the signal.
   * Call this on logout.
   */
  invalidate(): void {
    this._profile$ = null;
    this.profile.set(null);
  }

  /**
   * Forces a fresh fetch from the backend and updates the cache.
   * Call this after a profile update.
   */
  reload(): void {
    this._profile$ = null;
    this.profile$.subscribe({ error: () => undefined });
  }
}
