import { Injectable, inject } from '@angular/core';
import { map, Observable } from 'rxjs';

import { BackendApiService } from './backend-api.service';

@Injectable({ providedIn: 'root' })
export class AuthService {
  private readonly backendApi = inject(BackendApiService);

  login(email: string, password: string, rememberMe: boolean = false): Observable<string> {
    return this.backendApi.login(email, password, rememberMe).pipe(map((response) => response.message));
  }

  signup(email: string, password: string, displayName?: string): Observable<string> {
    return this.backendApi.signup(email, password, displayName).pipe(map((response) => response.message));
  }

  logout(): Observable<string> {
    return this.backendApi.logout().pipe(map((response) => response.message));
  }

  sendPasswordResetEmail(email: string): Observable<string> {
    return this.backendApi.sendPasswordResetEmail(email).pipe(map((response) => response.message));
  }

  updatePassword(password: string, accessToken: string): Observable<string> {
    return this.backendApi.updatePassword(password, accessToken).pipe(map((response) => response.message));
  }
}
