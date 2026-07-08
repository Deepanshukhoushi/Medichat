import { inject } from '@angular/core';
import { Router, CanActivateFn, CanActivateChildFn, ActivatedRouteSnapshot, RouterStateSnapshot } from '@angular/router';
import { catchError, map, of } from 'rxjs';
import { ProfileService } from '../services/profile.service';

/**
 * Routes that guests (unauthenticated users) are permitted to access.
 * Controlled declaratively via `data: { allowGuest: true }` in app.routes.ts
 * rather than by substring-matching the URL, so future routes cannot
 * accidentally inherit guest access.
 */
export const authGuard: CanActivateChildFn = (route: ActivatedRouteSnapshot, _state: RouterStateSnapshot) => {
  const profileService = inject(ProfileService);
  const router = inject(Router);
  const allowGuest: boolean = route.data?.['allowGuest'] === true;

  return profileService.profile$.pipe(
    map((profile) => {
      const isGuest = profile?.user_id?.startsWith('guest_');

      if (isGuest && !allowGuest) {
        return router.createUrlTree(['/auth/login']);
      }
      return true;
    }),
    catchError(() => {
      // No valid session — allow guests only on explicitly flagged routes.
      if (allowGuest) {
        return of(true);
      }
      return of(router.createUrlTree(['/auth/login']));
    })
  );
};
