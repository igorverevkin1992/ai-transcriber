// Triggers a browser download for a Blob and cleans up the object URL.
//
// revokeObjectURL is deferred so the browser has time to start the download
// before the URL is invalidated (revoking synchronously after click() can
// cancel the download in some browsers).
export function downloadBlob(blob: Blob, filename: string): void {
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    window.URL.revokeObjectURL(url);
    a.remove();
  }, 1000);
}
