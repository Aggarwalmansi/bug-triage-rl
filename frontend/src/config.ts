const DEFAULT_LOCAL_API = "http://127.0.0.1:8000";
const PROD_API_BASE = "https://wahhbhai-bug-triage-env.hf.space";

function trimTrailingSlash(value: string) {
  return value.replace(/\/+$/, "");
}

export function resolveApiBase() {
  const fromEnv = import.meta.env.VITE_API_URL;
  if (fromEnv) {
    return trimTrailingSlash(fromEnv);
  }

  if (typeof window !== "undefined") {
    if (window.location.port === "5173" || window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1") {
      return DEFAULT_LOCAL_API;
    }

    return PROD_API_BASE;
  }

  return DEFAULT_LOCAL_API;
}

export const API_BASE = resolveApiBase();
