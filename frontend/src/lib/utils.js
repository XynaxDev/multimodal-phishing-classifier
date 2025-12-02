export function cn(...inputs) {
  return inputs
    .filter(Boolean)
    .flatMap((input) =>
      typeof input === "string"
        ? input.split(" ")
        : Array.isArray(input)
        ? input
        : []
    )
    .join(" ");
}
