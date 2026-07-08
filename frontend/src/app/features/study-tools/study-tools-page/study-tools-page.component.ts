import { ChangeDetectionStrategy, Component, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpEventType } from '@angular/common/http';
import { FormBuilder, ReactiveFormsModule, Validators } from '@angular/forms';
import { MarkdownModule } from 'ngx-markdown';

import { GlassCardComponent } from '../../../shared/components/glass-card/glass-card.component';
import { SectionHeadingComponent } from '../../../shared/components/section-heading/section-heading.component';
import { BackendApiService } from '../../../core/services/backend-api.service';
import { extractErrorMessage } from '../../../core/utils/extract-error-message';

@Component({
  selector: 'mc-study-tools-page',
  standalone: true,
  imports: [GlassCardComponent, SectionHeadingComponent, CommonModule, ReactiveFormsModule, MarkdownModule],
  templateUrl: './study-tools-page.component.html',
  styleUrl: './study-tools-page.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class StudyToolsPageComponent {
  private api = inject(BackendApiService);
  private fb = inject(FormBuilder);

  explainForm = this.fb.nonNullable.group({ topic: ['', Validators.required] });
  summarizeForm = this.fb.nonNullable.group({ text: ['', Validators.required] });
  mnemonicsForm = this.fb.nonNullable.group({ topic: ['', Validators.required] });

  readonly explainResult = signal<string | null>(null);
  readonly summarizeResult = signal<string | null>(null);
  readonly mnemonicsResult = signal<string | null>(null);
  readonly explainError = signal<string | null>(null);
  readonly summarizeError = signal<string | null>(null);
  readonly mnemonicsError = signal<string | null>(null);
  readonly explainLoading = signal(false);
  readonly summarizeLoading = signal(false);
  readonly mnemonicsLoading = signal(false);

  readonly isUploading = signal(false);
  readonly uploadResult = signal<string | null>(null);
  readonly uploadError = signal<string | null>(null);
  readonly uploadPulse = signal(0);

  /** Must match backend max_upload_size_bytes (10 MB). */
  private readonly MAX_UPLOAD_BYTES = 10 * 1024 * 1024;

  explainTopic() {
    if (this.explainForm.invalid) return;
    this.explainLoading.set(true);
    this.explainError.set(null);
    this.api.explainTopic(this.explainForm.getRawValue().topic).subscribe({
      next: (res) => {
        this.explainResult.set(res.result);
        this.explainLoading.set(false);
      },
      error: (err) => {
        this.explainError.set(extractErrorMessage(err, 'Failed to explain topic. Please try again.'));
        this.explainLoading.set(false);
      }
    });
  }

  summarizeText() {
    if (this.summarizeForm.invalid) return;
    this.summarizeLoading.set(true);
    this.summarizeError.set(null);
    this.api.summarizeText(this.summarizeForm.getRawValue().text).subscribe({
      next: (res) => {
        this.summarizeResult.set(res.result);
        this.summarizeLoading.set(false);
      },
      error: (err) => {
        this.summarizeError.set(extractErrorMessage(err, 'Failed to summarize text. Please try again.'));
        this.summarizeLoading.set(false);
      }
    });
  }

  generateMnemonics() {
    if (this.mnemonicsForm.invalid) return;
    this.mnemonicsLoading.set(true);
    this.mnemonicsError.set(null);
    this.api.generateMnemonics(this.mnemonicsForm.getRawValue().topic).subscribe({
      next: (res) => {
        this.mnemonicsResult.set(res.result);
        this.mnemonicsLoading.set(false);
      },
      error: (err) => {
        this.mnemonicsError.set(extractErrorMessage(err, 'Failed to generate mnemonics. Please try again.'));
        this.mnemonicsLoading.set(false);
      }
    });
  }

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (!input.files || input.files.length === 0) return;

    const file = input.files[0];

    // --- Pre-flight validation (avoids burning bandwidth on a certain reject) ---
    const allowedExtensions = ['.pdf', '.docx', '.txt'];
    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!allowedExtensions.includes(fileExtension)) {
      this.uploadError.set(`Unsupported file type "${fileExtension}". Please upload a PDF, DOCX, or TXT file.`);
      input.value = '';
      return;
    }
    if (file.size > this.MAX_UPLOAD_BYTES) {
      const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
      this.uploadError.set(`File is too large (${sizeMB} MB). Maximum allowed size is 10 MB.`);
      input.value = '';
      return;
    }

    this.isUploading.set(true);
    this.uploadResult.set(null);
    this.uploadError.set(null);
    this.uploadPulse.set(0);
    this.api.uploadDocumentWithProgress(file).subscribe({
      next: (event) => {
        if (event.type === HttpEventType.UploadProgress) {
          const total = event.total ?? file.size;
          const progress = total > 0 ? Math.round((100 * event.loaded) / total) : 0;
          this.uploadPulse.set(Math.min(progress, 100));
          return;
        }

        if (event.type === HttpEventType.Response && event.body) {
          this.isUploading.set(false);
          this.uploadPulse.set(100);
          this.uploadResult.set(`Successfully indexed ${event.body.chunks_indexed} chunks from ${event.body.filename}.`);
          input.value = '';
          setTimeout(() => this.uploadPulse.set(0), 700);
        }
      },
      error: (err) => {
        this.isUploading.set(false);
        this.uploadPulse.set(0);
        this.uploadError.set(extractErrorMessage(err, 'Failed to upload document'));
        input.value = '';
      }
    });
  }
}
