import { ChangeDetectionStrategy, Component } from '@angular/core';
import { RouterLink } from '@angular/router';
import { LucideDynamicIcon } from '@lucide/angular';

import { MedicalOrbComponent } from '../../../shared/components/medical-orb/medical-orb.component';
import { RevealDirective } from '../../../shared/directives/reveal.directive';
import { appIcons } from '../../../shared/icons/lucide-icons';

@Component({
  selector: 'mc-landing-page',
  standalone: true,
  imports: [RouterLink, LucideDynamicIcon, MedicalOrbComponent, RevealDirective],
  templateUrl: './landing-page.component.html',
  styleUrl: './landing-page.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class LandingPageComponent {
  protected readonly icons = appIcons;
  protected readonly features = [
    {
      title: 'AI Chat',
      description: 'Ask follow-up questions and get clear explanations for complex medical topics.',
      icon: appIcons.MessagesSquare
    },
    {
      title: 'Smart Flashcards',
      description: 'Turn lecture notes into active-recall flashcards you can study anywhere.',
      icon: appIcons.WandSparkles
    },
    {
      title: 'Timed Quizzes',
      description: 'Check your knowledge with multiple-choice quizzes built around your study topics.',
      icon: appIcons.Clock3
    },
    {
      title: 'Document Upload',
      description: 'Upload PDFs, notes, or handouts and turn them into searchable study material.',
      icon: appIcons.Paperclip
    },
    {
      title: 'AI Study Tools',
      description: 'Summarize, explain, and generate mnemonics when you need a faster study pass.',
      icon: appIcons.Lightbulb
    },
    {
      title: 'Analytics',
      description: 'Track sessions, streaks, and learning trends so you can see consistent progress.',
      icon: appIcons.ChartColumnBig
    }
  ];
}
