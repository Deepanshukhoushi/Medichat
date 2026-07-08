import { CanDeactivateFn } from '@angular/router';
import { QuizzesPageComponent } from '../../features/quizzes/quizzes-page/quizzes-page.component';

export const quizGuard: CanDeactivateFn<QuizzesPageComponent> = (component) => {
  if (component.activeQuiz() && !component.quizSubmitted()) {
    const confirmation = window.confirm('You have an active quiz that is not submitted. Are you sure you want to leave and lose your progress?');
    if (!confirmation) {
      return false;
    }
  }
  return true;
};
