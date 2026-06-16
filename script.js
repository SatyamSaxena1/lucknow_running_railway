const filters = document.querySelectorAll(".filter");
const cards = document.querySelectorAll(".project-card");

filters.forEach((filter) => {
  filter.addEventListener("click", () => {
    const selected = filter.dataset.filter;

    filters.forEach((item) => {
      const isActive = item === filter;
      item.classList.toggle("active", isActive);
      item.setAttribute("aria-selected", String(isActive));
    });

    cards.forEach((card) => {
      const tags = card.dataset.tags.split(" ");
      const shouldShow = selected === "all" || tags.includes(selected);
      card.hidden = !shouldShow;
    });
  });
});
