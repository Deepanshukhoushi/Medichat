import { ChangeDetectionStrategy, Component, OnInit, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, Router, RouterLink } from '@angular/router';
import { ToastrService } from 'ngx-toastr';

import { AuthService } from '../../../core/services/auth.service';

@Component({
  selector: 'mc-reset-password-page',
  standalone: true,
  imports: [RouterLink, FormsModule],
  templateUrl: './reset-password-page.component.html',
  styleUrls: ['./reset-password-page.component.scss', '../auth-page.shared.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class ResetPasswordPageComponent implements OnInit {
  private readonly route = inject(ActivatedRoute);
  private readonly router = inject(Router);
  private readonly authService = inject(AuthService);
  private readonly toastr = inject(ToastrService);

  private accessToken = '';

  protected readonly password = signal('');
  protected readonly confirmPassword = signal('');
  protected readonly isSubmitting = signal(false);
  protected readonly statusMessage = signal('');
  protected readonly isTokenValid = signal(false);
  protected readonly isSuccess = signal(false);
  protected readonly submitted = signal(false);

  ngOnInit(): void {
    // Check fragment (hash) first
    this.route.fragment.subscribe((fragment) => {
      if (fragment) {
        const params = new URLSearchParams(fragment);
        this.parseParams(params);
      } else {
        // Fall back to query parameters
        this.route.queryParams.subscribe((queryParams) => {
          const params = new URLSearchParams();
          Object.keys(queryParams).forEach((key) => params.set(key, queryParams[key]));
          this.parseParams(params);
        });
      }
    });
  }

  private parseParams(params: URLSearchParams): void {
    const token = params.get('access_token');
    // Supabase sets type to recovery, but support direct token usage as well
    if (token) {
      this.accessToken = token;
      this.isTokenValid.set(true);
      this.statusMessage.set('');
    } else {
      this.isTokenValid.set(false);
      this.statusMessage.set('Invalid or missing password recovery parameters.');
    }
  }

  protected submit(): void {
    this.submitted.set(true);
    this.statusMessage.set('');
    if (!this.accessToken) {
      this.statusMessage.set('Invalid recovery token. Please request a new link.');
      return;
    }

    const pass = this.password();
    const conf = this.confirmPassword();

    if (!pass || pass.length < 6) {
      return;
    }

    if (pass !== conf) {
      return;
    }

    this.isSubmitting.set(true);

    this.authService.updatePassword(pass, this.accessToken).subscribe({
      next: async () => {
        this.isSuccess.set(true);
        this.isSubmitting.set(false);
        this.toastr.success('Password updated successfully. You can now log in.', 'Success');
        await this.router.navigateByUrl('/auth/login');
      },
      error: (error: { error?: { error?: string } }) => {
        this.statusMessage.set(error?.error?.error ?? 'Unable to update password right now.');
        this.isSubmitting.set(false);
      }
    });
  }
}
