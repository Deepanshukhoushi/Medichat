import { ChangeDetectionStrategy, Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { ToastrService } from 'ngx-toastr';

import { BackendApiService } from '../../../core/services/backend-api.service';
import { AuthService } from '../../../core/services/auth.service';
import { GoogleIconComponent } from '../../../shared/components/google-icon/google-icon.component';

@Component({
  selector: 'mc-login-page',
  standalone: true,
  imports: [FormsModule, RouterLink, GoogleIconComponent],
  templateUrl: './login-page.component.html',
  styleUrls: ['./login-page.component.scss', '../auth-page.shared.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class LoginPageComponent {
  private readonly authService = inject(AuthService);
  private readonly router = inject(Router);
  private readonly backendApi = inject(BackendApiService);
  private readonly toastr = inject(ToastrService);

  protected readonly email = signal('');
  protected readonly password = signal('');
  protected readonly rememberMe = signal(false);
  protected readonly isSubmitting = signal(false);
  protected readonly statusMessage = signal('');
  protected readonly statusType = signal<'success' | 'error' | null>(null);
  protected readonly submitted = signal(false);
  protected readonly showPassword = signal(false);
  protected readonly googleOAuthUrl = this.backendApi.googleLoginUrl();

  protected submit(): void {
    this.submitted.set(true);
    this.statusMessage.set('');
    this.statusType.set(null);
    if (!this.email().trim() || !this.password().trim()) {
      return;
    }
    
    this.isSubmitting.set(true);

    this.authService.login(this.email().trim(), this.password(), this.rememberMe()).subscribe({
      next: async (message: string) => {
        this.isSubmitting.set(false);
        this.toastr.success(message, 'Signed in');
        await this.router.navigateByUrl('/app/chat');
      },
      error: (error: { status?: number; error?: { error?: string; type?: string } }) => {
        const body = error?.error;
        const msg = body?.type === 'email_not_confirmed'
          ? (body.error ?? 'Please confirm your email before logging in.')
          : (body?.error ?? 'Unable to sign in right now.');
        this.statusMessage.set(msg);
        this.statusType.set('error');
        this.isSubmitting.set(false);
      }
    });
  }

  protected togglePasswordVisibility(): void {
    this.showPassword.update((value) => !value);
  }
}
