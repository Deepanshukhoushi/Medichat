import { inject } from '@angular/core';
import { Router, CanActivateFn } from '@angular/router';
import { catchError, map, of } from 'rxjs';
import { BackendApiService } from '../services/backend-api.service';

import { ActivatedRouteSnapshot, RouterStateSnapshot } from '@angular/router';

export const authGuard: CanActivateFn = (route: ActivatedRouteSnapshot, state: RouterStateSnapshot) => {
  const backendApi = inject(BackendApiService);
  const router = inject(Router);

  return backendApi.getProfile().pipe(
    map((profile) => {
      const isGuest = profile?.user_id?.startsWith('guest_');
      
      // Guests are allowed on chat and study-tools
      if (isGuest && !state.url.includes('/chat') && !state.url.includes('/study-tools')) {
        router.navigateByUrl('/auth/login');
        return false;
      }
      return true;
    }),
    catchError(() => {
      // Not authenticated or missing valid session, allow guests on chat
      if (state.url.includes('/chat')) {
        return of(true);
      }
      if (state.url.includes('/study-tools')) {
        return of(true);
      }
      router.navigateByUrl('/auth/login');
      return of(false);
    })
  );
};
