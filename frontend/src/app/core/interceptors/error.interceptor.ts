import { HttpErrorResponse, HttpInterceptorFn } from '@angular/common/http';
import { inject } from '@angular/core';
import { ToastrService } from 'ngx-toastr';
import { throwError } from 'rxjs';
import { catchError } from 'rxjs/operators';

export const errorInterceptor: HttpInterceptorFn = (req, next) => {
  const toastr = inject(ToastrService);

  return next(req).pipe(
    catchError((error: HttpErrorResponse) => {
      let errorMessage = 'An unexpected error occurred.';
      
      if (error.error instanceof ErrorEvent) {
        // Client-side error
        errorMessage = error.error.message;
      } else {
        // Server-side error
        if (error.error && error.error.error) {
          errorMessage = error.error.error;
        } else if (error.error && error.error.message) {
          errorMessage = error.error.message;
        } else if (error.status === 429) {
          errorMessage = 'Too many requests. Please try again later.';
        } else if (error.status === 0) {
          errorMessage = 'Unable to connect to the server. Check your network connection.';
        }
      }

      // Ignore 401s for guest logic / auth polling
      if (error.status !== 401 && error.status !== 404) {
        toastr.error(errorMessage, 'Error');
      }

      return throwError(() => error);
    })
  );
};
