import { Injectable } from '@angular/core';
import { environment } from '../../../environments/environment';

@Injectable({ providedIn: 'root' })
export class RuntimeConfigService {
  readonly apiBaseUrl = this.readApiBaseUrl();

  private readApiBaseUrl(): string {
    if (typeof document !== 'undefined') {
      const metaUrl = document.querySelector<HTMLMetaElement>('meta[name="medichat-api-base-url"]')?.content?.trim();
      if (metaUrl) {
        return metaUrl;
      }
    }

    const runtimeUrl = (globalThis as { __MEDICHAT_CONFIG__?: { apiBaseUrl?: string } }).__MEDICHAT_CONFIG__?.apiBaseUrl?.trim();
    if (runtimeUrl) {
      return runtimeUrl;
    }

    return environment.apiBaseUrl;
  }
}
