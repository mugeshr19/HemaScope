"use client";
import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, ImageIcon, X } from "lucide-react";
import { cn } from "@/lib/utils";

interface DropzoneProps {
  onFile: (file: File) => void;
  disabled?: boolean;
}

export function Dropzone({ onFile, disabled }: DropzoneProps) {
  const [preview, setPreview] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);

  const onDrop = useCallback(
    (accepted: File[]) => {
      if (accepted[0]) {
        setPreview(URL.createObjectURL(accepted[0]));
        setFileName(accepted[0].name);
        onFile(accepted[0]);
      }
    },
    [onFile]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"] },
    maxFiles: 1,
    disabled,
  });

  const clear = (e: React.MouseEvent) => {
    e.stopPropagation();
    setPreview(null);
    setFileName(null);
  };

  return (
    <div
      {...getRootProps()}
      className={cn(
        "relative border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors",
        isDragActive ? "border-primary bg-primary/5" : "border-border hover:border-primary/50 hover:bg-accent/30",
        disabled && "opacity-50 cursor-not-allowed"
      )}
    >
      <input {...getInputProps()} />
      {preview ? (
        <div className="relative">
          <img src={preview} alt="Preview" className="max-h-64 mx-auto rounded object-contain" />
          <button onClick={clear} className="absolute top-1 right-1 p-1 bg-black/60 rounded-full text-white hover:bg-black/80">
            <X className="w-4 h-4" />
          </button>
          <p className="text-sm text-muted-foreground mt-2">{fileName}</p>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-3">
          <div className="p-4 rounded-full bg-primary/10">
            {isDragActive ? <ImageIcon className="w-8 h-8 text-primary" /> : <Upload className="w-8 h-8 text-primary" />}
          </div>
          <div>
            <p className="font-medium">{isDragActive ? "Drop image here" : "Drag & drop blood smear image"}</p>
            <p className="text-sm text-muted-foreground mt-1">or click to browse · JPG, PNG, BMP, TIFF</p>
          </div>
        </div>
      )}
    </div>
  );
}
