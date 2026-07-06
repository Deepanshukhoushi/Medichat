import { ChangeDetectionStrategy, Component, OnInit, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { ToastrService } from 'ngx-toastr';

import { AuthService } from '../../../core/services/auth.service';
import { BackendApiService } from '../../../core/services/backend-api.service';
import { GoogleIconComponent } from '../../../shared/components/google-icon/google-icon.component';

@Component({
  selector: 'mc-signup-page',
  standalone: true,
  imports: [FormsModule, RouterLink, GoogleIconComponent],
  templateUrl: './signup-page.component.html',
  styleUrls: ['./signup-page.component.scss', '../auth-page.shared.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class SignupPageComponent implements OnInit {
  private readonly authService = inject(AuthService);
  private readonly backendApi = inject(BackendApiService);
  private readonly router = inject(Router);
  private readonly toastr = inject(ToastrService);

  protected readonly email = signal('');
  protected readonly displayName = signal('');
  protected readonly password = signal('');
  protected readonly confirmPassword = signal('');
  protected readonly isSubmitting = signal(false);
  protected readonly statusMessage = signal('');
  protected readonly statusType = signal<'success' | 'error' | null>(null);
  protected readonly submitted = signal(false);
  protected readonly showPassword = signal(false);
  protected readonly showConfirmPassword = signal(false);
  protected readonly googleOAuthUrl = this.backendApi.googleLoginUrl();

  ngOnInit(): void {
    this.resetFormState();
  }

  protected submit(): void {
    this.submitted.set(true);
    this.statusMessage.set('');
    this.statusType.set(null);
    const emailValue = this.email().trim();
    const displayNameValue = this.displayName().trim();
    const passwordValue = this.password();
    const confirmPasswordValue = this.confirmPassword();

    if (!emailValue || !displayNameValue || !passwordValue.trim() || !confirmPasswordValue.trim()) {
      return;
    }

    if (passwordValue !== confirmPasswordValue) {
      return;
    }
    
    this.isSubmitting.set(true);

    this.authService.signup(emailValue, passwordValue, displayNameValue).subscribe({
      next: async (message: string) => {
        this.isSubmitting.set(false);
        this.toastr.success(message, 'Account created');
        await this.router.navigateByUrl('/auth/login');
      },
      error: (error: { error?: { error?: string } }) => {
        this.statusMessage.set(error?.error?.error ?? 'Unable to create your account right now.');
        this.statusType.set('error');
        this.isSubmitting.set(false);
      }
    });
  }

  protected togglePasswordVisibility(): void {
    this.showPassword.update((value) => !value);
  }

  protected toggleConfirmPasswordVisibility(): void {
    this.showConfirmPassword.update((value) => !value);
  }

  protected passwordStrengthLabel(): 'Weak' | 'Medium' | 'Strong' {
    const score = this.passwordStrengthScore();
    if (score < 3) {
      return 'Weak';
    }
    if (score < 5) {
      return 'Medium';
    }
    return 'Strong';
  }

  protected passwordStrengthPercent(): number {
    return Math.min(100, this.passwordStrengthScore() * 20);
  }

  protected passwordStrengthClass(): string {
    const score = this.passwordStrengthScore();
    if (score < 3) {
      return 'bg-red-500';
    }
    if (score < 5) {
      return 'bg-warning';
    }
    return 'bg-success';
  }

  private passwordStrengthScore(): number {
    const value = this.password();
    let score = 0;

    if (value.length >= 8) {
      score += 2;
    } else if (value.length >= 6) {
      score += 1;
    }
    if (/[A-Z]/.test(value)) {
      score += 1;
    }
    if (/[a-z]/.test(value)) {
      score += 1;
    }
    if (/\d/.test(value)) {
      score += 1;
    }
    if (/[^A-Za-z0-9]/.test(value)) {
      score += 1;
    }

    return score;
  }

  private resetFormState(): void {
    this.email.set('');
    this.displayName.set('');
    this.password.set('');
    this.confirmPassword.set('');
    this.isSubmitting.set(false);
    this.statusMessage.set('');
    this.statusType.set(null);
    this.submitted.set(false);
    this.showPassword.set(false);
    this.showConfirmPassword.set(false);
  }
}
