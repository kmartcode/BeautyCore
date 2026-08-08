// Mobile nav toggle
const toggle = document.querySelector('.nav-toggle');
const navLinks = document.querySelector('.nav-links');

if (toggle && navLinks) {
  toggle.addEventListener('click', () => {
    navLinks.classList.toggle('open');
  });
}

// Services page: sidebar active link on scroll
const sidebarLinks = document.querySelectorAll('.sidebar-nav a');
if (sidebarLinks.length) {
  const sections = Array.from(sidebarLinks).map(a => document.querySelector(a.getAttribute('href')));

  const onScroll = () => {
    let current = '';
    sections.forEach(sec => {
      if (sec && window.scrollY >= sec.offsetTop - 120) {
        current = '#' + sec.id;
      }
    });
    sidebarLinks.forEach(a => {
      a.classList.toggle('active', a.getAttribute('href') === current);
    });
  };

  window.addEventListener('scroll', onScroll, { passive: true });
}

// Contact form submission feedback
const form = document.querySelector('.contact-form');
if (form) {
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const btn = form.querySelector('.btn-submit');
    btn.textContent = 'MESSAGE SENT';
    btn.style.background = '#7b2fa0';
    btn.style.color = '#fff';
    setTimeout(() => {
      btn.textContent = 'SUBMIT INQUIRY';
      btn.style.background = '';
      btn.style.color = '';
      form.reset();
    }, 3000);
  });
}
