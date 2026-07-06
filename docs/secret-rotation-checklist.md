# Secret Rotation Checklist

Use this checklist whenever a credential may have been committed or copied into a repository file.

## Immediate

- Rotate `PINECONE_API_KEY` in the Pinecone console.
- Rotate `COHERE_API_KEY` in the Cohere dashboard.
- Rotate `SUPABASE_KEY` in the Supabase project settings.
- Rotate any Flask secret or session signing secret used in production.

## Git Hygiene

- Keep secrets only in a local `.env` file that is ignored by git.
- Confirm `.env.example` contains placeholders only.
- Review `git log` for historical commits that may contain credentials.
- If history exposure is confirmed, rewrite history in a coordinated maintenance window.

## Access Review

- Revoke any unused API keys.
- Verify deploy environments and CI variables are updated.
- Rotate tokens used by teammates or automation if they may have been copied.

