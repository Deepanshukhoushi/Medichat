import { ChangeDetectionStrategy, Component, inject, input, output, signal } from '@angular/core';
import { HttpEventType } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { LucideDynamicIcon } from '@lucide/angular';

import { BackendApiService } from '../../../../core/services/backend-api.service';
import { appIcons } from '../../../../shared/icons/lucide-icons';
import { extractErrorMessage } from '../../../../core/utils/extract-error-message';

@Component({
  selector: 'mc-chat-composer',
  standalone: true,
  imports: [FormsModule, LucideDynamicIcon],
  templateUrl: './chat-composer.component.html',
  styleUrl: './chat-composer.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class ChatComposerComponent {
  readonly submitted = output<string>();
  readonly stopped = output<void>();
  readonly focusStateChange = output<boolean>();
  readonly isStreaming = input(false);

  protected readonly icons = appIcons;
  protected readonly prompt = signal('');
  protected readonly isUploading = signal(false);
  protected readonly uploadProgress = signal(0);
  protected readonly uploadFileName = signal<string | null>(null);
  protected readonly uploadToast = signal<string | null>(null);
  protected readonly voiceToast = signal(false);

  private readonly api = inject(BackendApiService);

  protected send(): void {
    const value = this.prompt().trim();
    if (!value) return;
    this.submitted.emit(value);
    this.prompt.set('');
  }

  protected showVoiceToast(): void {
    this.voiceToast.set(true);
    setTimeout(() => this.voiceToast.set(false), 3000);
  }

  public setPrompt(text: string): void {
    this.prompt.set(text);
  }

  protected onKeyDown(event: KeyboardEvent, textarea: HTMLTextAreaElement): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      if (!this.isStreaming()) {
        this.send();
      }
      return;
    }

    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
      event.preventDefault();
      this.prompt.set('');
      textarea.focus();
      return;
    }

    if (event.key === 'Escape') {
      event.preventDefault();
      this.prompt.set('');
    }
  }

  protected triggerFileUpload(fileInput: HTMLInputElement): void {
    fileInput.click();
  }

  protected handleFocusOut(event: FocusEvent): void {
    const currentTarget = event.currentTarget as HTMLElement | null;
    const relatedTarget = event.relatedTarget as Node | null;
    if (currentTarget && relatedTarget && currentTarget.contains(relatedTarget)) {
      return;
    }

    this.focusStateChange.emit(false);
  }

  protected onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (!input.files || input.files.length === 0) return;

    const file = input.files[0];
    this.isUploading.set(true);
    this.uploadProgress.set(0);
    this.uploadFileName.set(file.name);
    this.uploadToast.set(null);

    this.api.uploadDocumentWithProgress(file).subscribe({
      next: (event) => {
        if (event.type === HttpEventType.UploadProgress) {
          const total = event.total ?? file.size;
          const progress = total > 0 ? Math.round((100 * event.loaded) / total) : 0;
          this.uploadProgress.set(Math.min(progress, 100));
          return;
        }

        if (event.type === HttpEventType.Response && event.body) {
          this.isUploading.set(false);
          this.uploadProgress.set(100);
          this.uploadToast.set(`Indexed ${event.body.chunks_indexed} chunks from "${event.body.filename}"`);
          setTimeout(() => this.uploadToast.set(null), 4000);
          input.value = '';
          setTimeout(() => this.uploadProgress.set(0), 700);
          this.uploadFileName.set(null);
        }
      },
      error: (err) => {
        this.isUploading.set(false);
        this.uploadProgress.set(0);
        this.uploadFileName.set(null);
        this.uploadToast.set(extractErrorMessage(err, 'Failed to upload file'));
        setTimeout(() => this.uploadToast.set(null), 4000);
        input.value = '';
      }
    });
  }
}
