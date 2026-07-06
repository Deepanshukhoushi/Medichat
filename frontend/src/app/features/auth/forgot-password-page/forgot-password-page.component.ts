import { ChangeDetectionStrategy, Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { RouterLink } from '@angular/router';

import { AuthService } from '../../../core/services/auth.service';

@Component({
  selector: 'mc-forgot-password-page',
  standalone: true,
  imports: [RouterLink, FormsModule],
  templateUrl: './forgot-password-page.component.html',
  styleUrls: ['./forgot-password-page.component.scss', '../auth-page.shared.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class ForgotPasswordPageComponent {
  private readonly authService = inject(AuthService);

  protected readonly email = signal('');
  protected readonly isSubmitting = signal(false);
  protected readonly statusMessage = signal('');
  protected readonly statusType = signal<'success' | 'error' | null>(null);
  protected readonly submitted = signal(false);

  protected submit(): void {
    this.submitted.set(true);
    this.statusMessage.set('');
    this.statusType.set(null);
    const emailVal = this.email().trim();
    if (!emailVal) {
      return;
    }

    this.isSubmitting.set(true);

    this.authService.sendPasswordResetEmail(emailVal).subscribe({
      next: (message: string) => {
        this.statusMessage.set(message || 'Password reset link sent to your email.');
        this.statusType.set('success');
        this.isSubmitting.set(false);
      },
      error: (error: { error?: { error?: string } }) => {
        this.statusMessage.set(error?.error?.error ?? 'Unable to send password reset email right now.');
        this.statusType.set('error');
        this.isSubmitting.set(false);
      }
    });
  }
}
