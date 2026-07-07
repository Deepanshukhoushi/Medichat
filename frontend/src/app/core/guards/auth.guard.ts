import { inject } from '@angular/core';
import { Router, CanActivateFn, ActivatedRouteSnapshot, RouterStateSnapshot } from '@angular/router';
import { catchError, map, of } from 'rxjs';
import { BackendApiService } from '../services/backend-api.service';

/**
 * Routes that guests (unauthenticated users) are permitted to access.
 * Controlled declaratively via `data: { allowGuest: true }` in app.routes.ts
 * rather than by substring-matching the URL, so future routes cannot
 * accidentally inherit guest access.
 */
export const authGuard: CanActivateFn = (route: ActivatedRouteSnapshot, _state: RouterStateSnapshot) => {
  const backendApi = inject(BackendApiService);
  const router = inject(Router);
  const allowGuest: boolean = route.data?.['allowGuest'] === true;

  return backendApi.getProfile().pipe(
    map((profile) => {
      const isGuest = profile?.user_id?.startsWith('guest_');

      if (isGuest && !allowGuest) {
        router.navigateByUrl('/auth/login');
        return false;
      }
      return true;
    }),
    catchError(() => {
      // No valid session — allow guests only on explicitly flagged routes.
      if (allowGuest) {
        return of(true);
      }
      router.navigateByUrl('/auth/login');
      return of(false);
    })
  );
};
