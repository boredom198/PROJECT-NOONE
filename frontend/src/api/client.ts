declare global {
  interface Window {
    __NOONE_TOKEN__?: string;
  }
}

function token(): string {
  return window.__NOONE_TOKEN__ || sessionStorage.getItem("noone_token") || "";
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);

  headers.set("Content-Type", "application/json");
  headers.set("X-NOONE-Token", token());

  const res = await fetch(path, {
    ...init,
    headers,
  });

  const data = (await res.json().catch(() => ({}))) as T & {
    ok?: boolean;
    error?: string;
  };

  if (!res.ok || data.ok === false) {
    throw new Error(data.error || `Request failed (${res.status})`);
  }

  return data as T;
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body: unknown) =>
    request<T>(path, { method: "POST", body: JSON.stringify(body) }),
};

export function fmt(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function fmtPrice(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  const digits = value >= 1000 ? 2 : value >= 1 ? 4 : 8;
  return fmt(value, digits);
}

export function fmtPct(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}%`;
}
