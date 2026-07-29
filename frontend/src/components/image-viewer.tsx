"use client";
import { useState } from "react";
import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";
import { ZoomIn, ZoomOut, RotateCcw } from "lucide-react";

interface ImageViewerProps {
  src: string;
  alt?: string;
}

export function ImageViewer({ src, alt = "Blood smear" }: ImageViewerProps) {
  return (
    <div className="relative rounded-lg overflow-hidden border border-border bg-black/20">
      <TransformWrapper initialScale={1} minScale={0.5} maxScale={8}>
        {({ zoomIn, zoomOut, resetTransform }) => (
          <>
            <div className="absolute top-2 right-2 z-10 flex gap-1">
              {[
                { icon: ZoomIn, action: () => zoomIn() },
                { icon: ZoomOut, action: () => zoomOut() },
                { icon: RotateCcw, action: () => resetTransform() },
              ].map(({ icon: Icon, action }, i) => (
                <button
                  key={i}
                  onClick={action}
                  className="p-1.5 rounded bg-black/60 text-white hover:bg-black/80 transition-colors"
                >
                  <Icon className="w-4 h-4" />
                </button>
              ))}
            </div>
            <TransformComponent wrapperClass="w-full" contentClass="w-full">
              <img src={src} alt={alt} className="w-full h-auto max-h-[500px] object-contain" />
            </TransformComponent>
          </>
        )}
      </TransformWrapper>
    </div>
  );
}
