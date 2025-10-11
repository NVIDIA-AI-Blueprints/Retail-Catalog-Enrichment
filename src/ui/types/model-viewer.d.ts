declare global {
  namespace JSX {
    interface IntrinsicElements {
      'model-viewer': React.DetailedHTMLProps<
        React.HTMLAttributes<HTMLElement> & {
          src?: string;
          alt?: string;
          'camera-controls'?: string | boolean;
          'auto-rotate'?: string | boolean;
          'shadow-intensity'?: string;
          'exposure'?: string;
        },
        HTMLElement
      >;
    }
  }
}

export {};

