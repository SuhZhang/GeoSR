document.addEventListener('DOMContentLoaded', () => {
  const burgers = Array.from(document.querySelectorAll('.navbar-burger'));

  burgers.forEach((burger) => {
    burger.addEventListener('click', () => {
      const target = burger.dataset.target;
      const menu = document.getElementById(target);

      burger.classList.toggle('is-active');
      if (menu) {
        menu.classList.toggle('is-active');
      }
    });
  });

  const copyButton = document.getElementById('copy-bibtex');
  const copyStatus = document.getElementById('copy-status');
  const bibtexBlock = document.getElementById('bibtex-block');

  if (copyButton && copyStatus && bibtexBlock) {
    copyButton.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(bibtexBlock.textContent);
        copyStatus.textContent = 'BibTeX copied.';
      } catch (error) {
        copyStatus.textContent = 'Clipboard unavailable. Copy manually from the block.';
      }
    });
  }
});
