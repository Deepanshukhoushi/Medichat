import { HttpContextToken } from '@angular/common/http';

/**
 * HTTP context token that suppresses the global error toast for a specific request.
 *
 * Usage (in components/services that show their own inline error):
 *
 *   import { HttpContext } from '@angular/common/http';
 *   import { SKIP_GLOBAL_ERROR } from './skip-global-error.token';
 *
 *   this.http.get('/api/...', {
 *     context: new HttpContext().set(SKIP_GLOBAL_ERROR, true)
 *   });
 *
 * The error.interceptor checks this token and skips toastr.error() when true.
 */
export const SKIP_GLOBAL_ERROR = new HttpContextToken<boolean>(() => false);
