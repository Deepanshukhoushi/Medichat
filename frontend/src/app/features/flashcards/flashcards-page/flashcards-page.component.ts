import { ChangeDetectionStrategy, Component, inject, OnInit, signal } from '@angular/core';
import { AsyncPipe, CommonModule } from '@angular/common';
import { FormBuilder, ReactiveFormsModule, Validators } from '@angular/forms';
import { BehaviorSubject } from 'rxjs';

import { GlassCardComponent } from '../../../shared/components/glass-card/glass-card.component';
import { SectionHeadingComponent } from '../../../shared/components/section-heading/section-heading.component';
import { BackendApiService, FlashcardDeck } from '../../../core/services/backend-api.service';
import { extractErrorMessage } from '../../../core/utils/extract-error-message';

@Component({
  selector: 'mc-flashcards-page',
  standalone: true,
  imports: [GlassCardComponent, SectionHeadingComponent, CommonModule, ReactiveFormsModule],
  templateUrl: './flashcards-page.component.html',
  styleUrl: './flashcards-page.component.scss',
  changeDetection: ChangeDetectionStrategy.OnPush
})
export class FlashcardsPageComponent implements OnInit {
  private api = inject(BackendApiService);
  private fb = inject(FormBuilder);

  decks$ = new BehaviorSubject<FlashcardDeck[]>([]);
  readonly isLoadingDecks = signal(false);
  readonly isGenerating = signal(false);
  readonly isLoadingDeck = signal(false);
  readonly errorMessage = signal<string | null>(null);
  readonly cardRatings = signal<Record<string, 'known' | 'unknown'>>({});

  form = this.fb.nonNullable.group({
    topic: ['', Validators.required],
    count: [5, [Validators.required, Validators.min(1), Validators.max(20)]]
  });

  ngOnInit() {
    this.loadDecks(true);
  }

  loadDecks(initial = false) {
    if (initial) {
      this.isLoadingDecks.set(true);
    }
    this.api.getFlashcardDecks().subscribe({
      next: (decks) => {
        this.decks$.next(decks);
        this.isLoadingDecks.set(false);
      },
      error: (err) => {
        console.error('Failed to load decks', err);
        this.isLoadingDecks.set(false);
      }
    });
  }

  generateDeck() {
    if (this.form.invalid) return;
    this.isGenerating.set(true);
    this.errorMessage.set(null);
    const { topic, count } = this.form.getRawValue();
    
    this.api.generateFlashcardDeck(topic, count).subscribe({
      next: () => {
        this.isGenerating.set(false);
        this.form.reset({ topic: '', count: 5 });
        this.loadDecks();
      },
      error: (err) => {
        console.error('Failed to generate deck', err);
        this.errorMessage.set(extractErrorMessage(err, 'Failed to generate deck. Please try again.'));
        this.isGenerating.set(false);
      }
    });
  }

  readonly activeDeck = signal<FlashcardDeck | null>(null);
  readonly currentCardIndex = signal(0);
  readonly isFlipped = signal(false);

  studyDeck(deckId: string) {
    this.isLoadingDeck.set(true);
    this.api.getFlashcardDeck(deckId).subscribe({
      next: (deck) => {
        this.activeDeck.set({
          ...deck,
          cards: this.shuffleCards(deck.cards ?? [])
        });
        this.currentCardIndex.set(0);
        this.isFlipped.set(false);
        this.cardRatings.set({});
        this.isLoadingDeck.set(false);
      },
      error: (err) => {
        console.error('Failed to load deck', err);
        this.isLoadingDeck.set(false);
      }
    });
  }

  flipCard() {
    this.isFlipped.update(v => !v);
  }

  handleCardKeydown(event: KeyboardEvent) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      this.flipCard();
      return;
    }
    if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
      event.preventDefault();
      this.nextCard();
      return;
    }
    if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
      event.preventDefault();
      this.prevCard();
    }
  }

  rateCurrentCard(rating: 'known' | 'unknown') {
    const deck = this.activeDeck();
    const cards = deck?.cards ?? [];
    const currentIndex = this.currentCardIndex();
    const currentCard = cards[currentIndex];

    if (!deck || !currentCard) {
      return;
    }

    // Key by card.front (stable card identity) so ratings survive array reshuffling.
    const cardKey = currentCard.front;
    this.cardRatings.update((current) => ({ ...current, [cardKey]: rating }));

    const remainingCards = cards.filter((_, index) => index !== currentIndex);
    const nextCards = [...remainingCards];
    const insertIndex = rating === 'known' ? nextCards.length : Math.min(1, nextCards.length);
    nextCards.splice(insertIndex, 0, currentCard);

    this.activeDeck.set({
      ...deck,
      cards: nextCards
    });
    this.currentCardIndex.set(Math.min(currentIndex, Math.max(nextCards.length - 1, 0)));
    this.isFlipped.set(false);

    if (currentCard.id) {
      this.api.rateFlashcard(deck.id, currentCard.id, rating).subscribe({
        error: (err) => console.error('Failed to save flashcard rating', err)
      });
    }
  }

  nextCard() {
    const deck = this.activeDeck();
    if (deck && deck.cards && this.currentCardIndex() < deck.cards.length - 1) {
      this.currentCardIndex.update(i => i + 1);
      this.isFlipped.set(false);
    }
  }

  prevCard() {
    if (this.currentCardIndex() > 0) {
      this.currentCardIndex.update(i => i - 1);
      this.isFlipped.set(false);
    }
  }

  closeDeck() {
    this.activeDeck.set(null);
  }

  private shuffleCards(cards: NonNullable<FlashcardDeck['cards']>): NonNullable<FlashcardDeck['cards']> {
    const shuffled = [...cards];
    for (let i = shuffled.length - 1; i > 0; i -= 1) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
  }
}
