import { HttpErrorResponse, HttpInterceptorFn } from '@angular/common/http';
import { inject } from '@angular/core';
import { ToastrService } from 'ngx-toastr';
import { throwError } from 'rxjs';
import { catchError } from 'rxjs/operators';
import { SKIP_GLOBAL_ERROR } from './skip-global-error.token';
import { extractErrorMessage } from '../utils/extract-error-message';

export const errorInterceptor: HttpInterceptorFn = (req, next) => {
  const toastr = inject(ToastrService);

  return next(req).pipe(
    catchError((error: HttpErrorResponse) => {
      const message = extractErrorMessage(error, 'An unexpected error occurred.');

      // Ignore 401s (guest auth polling) and 404s.
      // Also skip if the caller opted out with SKIP_GLOBAL_ERROR (they show their own inline banner).
      const skipGlobal = req.context.get(SKIP_GLOBAL_ERROR);
      if (error.status !== 401 && error.status !== 404 && !skipGlobal) {
        toastr.error(message, 'Error');
      }

      return throwError(() => error);
    })
  );
};
