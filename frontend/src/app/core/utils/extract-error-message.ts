import { HttpErrorResponse } from '@angular/common/http';

/**
 * Extracts a human-readable error message from an HTTP error response.
 *
 * Resolution order (matches what the backend actually sends):
 *   1. error.error.error  — Flask AppError / { "error": "message" }
 *   2. error.error.message — some Flask extensions use { "message": "..." }
 *   3. Status-code-specific strings for common cases (429, 0)
 *   4. fallback — caller-supplied default
 *
 * Using this helper in all subscribe({ error }) blocks ensures that
 * the inline banner and the global toast always show identical text for
 * the same failure, regardless of which backend error shape was returned.
 *
 * @param err     The raw error from the Observable error channel.
 * @param fallback A caller-supplied default shown when no message is parseable.
 */
export function extractErrorMessage(err: unknown, fallback: string): string {
  if (err instanceof HttpErrorResponse) {
    // Server-sent structured error body
    if (err.error && typeof err.error === 'object') {
      if (typeof err.error.error === 'string' && err.error.error) {
        return err.error.error;
      }
      if (typeof err.error.message === 'string' && err.error.message) {
        return err.error.message;
      }
    }
    // Client-side / network error
    if (err.error instanceof ErrorEvent && err.error.message) {
      return err.error.message;
    }
    // Status-code shorthands
    if (err.status === 429) {
      return 'Too many requests. Please try again later.';
    }
    if (err.status === 0) {
      return 'Unable to connect to the server. Check your network connection.';
    }
  }
  return fallback;
}
