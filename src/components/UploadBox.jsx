import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, X, File } from 'lucide-react';

export default function UploadBox({ 
  label, 
  accept, 
  onFileSelect, 
  selectedFile,
  onRemove 
}) {
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      onFileSelect(acceptedFiles[0]);
    }
  }, [onFileSelect]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: accept ? { [accept]: [] } : undefined,
    multiple: false,
  });

  return (
    <div className="w-full">
      <label className="block text-sm font-semibold text-gray-700 mb-2">
        {label}
      </label>
      {selectedFile ? (
        <div className="flex items-center justify-between p-4 bg-primary-50 border-2 border-primary-200 rounded-lg">
          <div className="flex items-center space-x-3">
            <File size={20} className="text-primary-600" />
            <span className="text-sm text-gray-700">{selectedFile.name}</span>
          </div>
          <button
            onClick={onRemove}
            className="p-1 hover:bg-primary-100 rounded transition-colors"
          >
            <X size={18} className="text-gray-600" />
          </button>
        </div>
      ) : (
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all ${
            isDragActive
              ? 'border-primary-500 bg-primary-50'
              : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50'
          }`}
        >
          <input {...getInputProps()} />
          <Upload size={32} className="mx-auto mb-3 text-gray-400" />
          <p className="text-sm text-gray-600 mb-1">
            {isDragActive ? 'Drop file here' : 'Drag & drop or click to upload'}
          </p>
          <p className="text-xs text-gray-500">
            {accept || 'Any file type'}
          </p>
        </div>
      )}
    </div>
  );
}

