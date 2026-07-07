import { inject } from '@angular/core';
import { CanActivateFn, Router } from '@angular/router';
import { catchError, map, of } from 'rxjs';
import { BackendApiService } from '../services/backend-api.service';

/**
 * Prevents already-authenticated users from landing on auth pages
 * (login, signup, forgot-password, reset-password).
 *
 * If getProfile() succeeds and the user is NOT a guest → redirect to /app/chat.
 * If the user is a guest, or the call fails (no session), → allow through normally.
 *
 * This is the inverse of authGuard and is applied to the entire 'auth' route group.
 */
export const alreadyAuthGuard: CanActivateFn = () => {
  const backendApi = inject(BackendApiService);
  const router = inject(Router);

  return backendApi.getProfile().pipe(
    map((profile) => {
      const isGuest = profile?.user_id?.startsWith('guest_');
      if (!isGuest) {
        // Fully authenticated — send them to the app instead of the auth form.
        router.navigateByUrl('/app/chat');
        return false;
      }
      // Guest session — allow them to log in / sign up.
      return true;
    }),
    catchError(() => {
      // No session at all — the auth pages are exactly where they should be.
      return of(true);
    })
  );
};
