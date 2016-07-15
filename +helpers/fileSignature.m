function signature = fileSignature(varargin)
% FILESIGNATURE Computes a signature for a file
%   SIGNATURE = FILESIGNATURE(FILEPATH) computes signature for the
%   file FILEPATH based on its name and modification date. This is
%   meant as a unique file identifier for hashing purposes.
%
%   SIGNATURE = FILESIGNATRUE(FILE1PATH, FILEPATH2, ...) concatenates the
%   signatures of the specified files.
%
%   In more detail, he SIGNATURE is obtained as the string
%   "file_name;modification_date;". The file_name is without path and
%   the modification date is in the format returned by the MATLAB dir
%   function (i.e. a date field).

% Authors: Karel Lenc, Andrea Vedaldi

% AUTORIGHTS

% MODIFICATION !!!! Added the directory name on the file Signature, since
% there are datasets (Fischer and SymBench) whose images share the same
% name, date and hour, generating the same signature, even when they are
% form different datasets.
%
% The directory name is hence added to the signature

signature = '';

for i = 1:nargin
  f_info = dir(varargin{i});
  
  % ADDED: Get the 'sequence' name
  fragments = strsplit(varargin{i},'/');
  seqName = fragments{end-1};

  signature = strcat(signature,sprintf('%s;%s;%s;', seqName, f_info.name, f_info.date));
  %signature = strcat(signature,sprintf('%s;%s;', f_info.name, f_info.date));

end

