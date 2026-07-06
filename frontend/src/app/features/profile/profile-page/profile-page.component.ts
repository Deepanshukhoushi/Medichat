import { ChangeDetectionStrategy, Component, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { finalize } from 'rxjs';

import { GlassCardComponent } from '../../../shared/components/glass-card/glass-card.component';
import { SectionHeadingComponent } from '../../../shared/components/section-heading/section-heading.component';
import { BackendApiService, UserProfile } from '../../../core/services/backend-api.service';

@Component({
  selector: 'mc-profile-page',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule, GlassCardComponent, SectionHeadingComponent],
  templateUrl: './profile-page.component.html',
  styleUrl: './profile-page.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class ProfilePageComponent {
  private readonly backendApi = inject(BackendApiService);
  private readonly fb = inject(FormBuilder);

  readonly isLoading = signal(true);
  readonly isSaving = signal(false);
  readonly successMessage = signal<string | null>(null);
  readonly errorMsg = signal<string | null>(null);

  form: FormGroup = this.fb.group({
    display_name: ['', Validators.required],
    medical_year: [null],
    specialty: [''],
    university: ['']
  });

  constructor() {
    this.loadProfile();
  }

  loadProfile(): void {
    this.backendApi.getProfile().pipe(
      finalize(() => this.isLoading.set(false))
    ).subscribe({
      next: (profile) => {
        this.form.patchValue({
          display_name: profile.display_name || '',
          medical_year: profile.medical_year || null,
          specialty: profile.specialty || '',
          university: profile.university || ''
        });
      },
      error: () => {
        this.errorMsg.set('Failed to load profile. Please refresh the page.');
        setTimeout(() => this.errorMsg.set(null), 5000);
      }
    });
  }

  saveProfile(): void {
    if (this.form.invalid) return;

    this.isSaving.set(true);
    this.successMessage.set(null);
    this.errorMsg.set(null);

    const data: Partial<UserProfile> = this.form.value;

    this.backendApi.updateProfile(data).pipe(
      finalize(() => this.isSaving.set(false))
    ).subscribe({
      next: () => {
        this.successMessage.set('Profile updated successfully!');
        setTimeout(() => this.successMessage.set(null), 3000);
      },
      error: (err) => {
        this.errorMsg.set(err?.error?.error || 'Failed to save profile. Please try again.');
        setTimeout(() => this.errorMsg.set(null), 5000);
      }
    });
  }
}
