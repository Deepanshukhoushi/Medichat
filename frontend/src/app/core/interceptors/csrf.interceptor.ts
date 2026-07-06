import { HttpInterceptorFn } from '@angular/common/http';
import { inject } from '@angular/core';
import { from, switchMap } from 'rxjs';
import { catchError } from 'rxjs/operators';

import { RuntimeConfigService } from '../services/runtime-config.service';

function readCookie(name: string): string | null {
  if (typeof document === 'undefined') {
    return null;
  }

  const match = document.cookie.split('; ').find((entry) => entry.startsWith(`${name}=`));
  return match ? decodeURIComponent(match.split('=').slice(1).join('=')) : null;
}

function buildBackendUrl(baseUrl: string, path: string): string {
  const trimmedBase = baseUrl.replace(/\/+$/, '');
  const trimmedPath = path.startsWith('/') ? path : `/${path}`;
  return `${trimmedBase}${trimmedPath}`;
}

export const csrfInterceptor: HttpInterceptorFn = (req, next) => {
  const runtimeConfig = inject(RuntimeConfigService);
  const apiMethods = new Set(['POST', 'PUT', 'PATCH', 'DELETE']);

  if (apiMethods.has(req.method)) {
    let token = readCookie('csrf_token');
    
    if (!token && typeof window !== 'undefined') {
      // Use fetch directly to avoid interceptor loops and ensure cookies are set
      return from(
        fetch(buildBackendUrl(runtimeConfig.apiBaseUrl, '/api/profile/me'), {
          method: 'GET',
          credentials: 'include',
        }).then(() => {
          token = readCookie('csrf_token');
          return token ?? '';
        })
      ).pipe(
        switchMap((csrfToken) => {
          const clonedRequest = req.clone({
            withCredentials: true,
            setHeaders: {
              'X-CSRF-Token': csrfToken
            }
          });
          return next(clonedRequest);
        }),
        catchError((err) => {
          console.error('CSRF token fetch failed:', err);
          const clonedRequest = req.clone({
            withCredentials: true,
            setHeaders: {
              'X-CSRF-Token': ''
            }
          });
          return next(clonedRequest);
        })
      );
    }

    const clonedRequest = req.clone({
      withCredentials: true,
      setHeaders: {
        'X-CSRF-Token': token ?? ''
      }
    });
    return next(clonedRequest);
  }

  return next(req.clone({ withCredentials: true }));
};
