import { ChangeDetectionStrategy, Component, OnInit, computed, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router, RouterLink } from '@angular/router';
import { LucideDynamicIcon } from '@lucide/angular';

import { RevealDirective } from '../../../shared/directives/reveal.directive';
import { CountUpDirective } from '../../../shared/directives/count-up.directive';
import { appIcons } from '../../../shared/icons/lucide-icons';
import { ProfileService } from '../../../core/services/profile.service';

@Component({
  selector: 'mc-landing-page',
  standalone: true,
  imports: [RouterLink, LucideDynamicIcon, CommonModule, RevealDirective, CountUpDirective],
  templateUrl: './landing-page.component.html',
  styleUrl: './landing-page.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class LandingPageComponent implements OnInit {
  private readonly profileService = inject(ProfileService);
  private readonly router = inject(Router);

  protected readonly isLoggedIn = computed(() => {
    const profile = this.profileService.profile();
    return profile?.user_id != null && !profile.user_id.startsWith('guest_');
  });
  protected readonly activeFaq = signal<number | null>(null);

  ngOnInit(): void {
    // Warm up the shared cache (guard likely already did this).
    this.profileService.profile$.subscribe({ error: () => {} });
  }

  protected toggleFaq(index: number) {
    this.activeFaq.update(current => current === index ? null : index);
  }

  protected readonly icons = appIcons;
  
  protected readonly features = [
    { title: 'AI Chat', description: 'Ask complex questions, get clear explanations backed by trusted medical textbooks.', icon: appIcons.MessagesSquare },
    { title: 'Smart Flashcards', description: 'Generate active-recall flashcards instantly from your current study session.', icon: appIcons.Layers },
    { title: 'Quiz Generator', description: 'Test your knowledge with multiple-choice questions tailored to your weaknesses.', icon: appIcons.ClipboardList },
    { title: 'Study Planner', description: 'Organize your topics, set goals, and track your learning streak every day.', icon: appIcons.Calendar },
    { title: 'Document Upload', description: 'Upload your class PDFs or notes to chat with them directly.', icon: appIcons.FileText },
    { title: 'Conversation History', description: 'Never lose a good explanation. All your past chats are safely stored and searchable.', icon: appIcons.History },
    { title: 'Streaming AI', description: 'Experience lightning-fast streaming responses so you never have to wait to learn.', icon: appIcons.Zap },
    { title: 'Bookmarks', description: 'Save specific messages, definitions, or mnemonics to review them later.', icon: appIcons.Bookmark },
    { title: 'Medical Images', description: 'Analyze anatomical diagrams and radiology images with vision AI.', icon: appIcons.Image }
  ];

  protected readonly faqs = [
    { q: 'Is MediChat free to use?', a: 'Yes, MediChat offers a generous free tier for medical students to get started.' },
    { q: 'Can I trust the AI answers?', a: 'Our AI is specifically tuned on medical literature and encourages you to cross-reference important clinical information. However, it is an educational tool, not a diagnostic one.' },
    { q: 'Does it work for nursing or pre-med?', a: 'Absolutely. MediChat adapts its complexity based on the questions you ask, making it perfect for MBBS, nursing, BDS, and pre-med students.' },
    { q: 'Can I upload my own lectures?', a: 'Yes! You can upload your PDF handouts and the AI will answer questions directly from your own material.' }
  ];

  protected readonly testimonials = [
    { quote: "MediChat completely changed how I study for anatomy. The instant flashcards save me hours.", author: "Sarah J.", role: "Med Student" },
    { quote: "The ability to upload my lecture slides and just ask questions is mind-blowing. It's like a 24/7 tutor.", author: "Michael T.", role: "Pre-med" },
    { quote: "I use the quiz generator before every exam. It finds the exact weak spots I didn't know I had.", author: "Dr. Emily R.", role: "Resident" },
    { quote: "Finally, an AI that understands medical terminology accurately without hallucinating random facts.", author: "James K.", role: "Nursing Student" },
  ];
}
