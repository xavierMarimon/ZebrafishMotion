function [marker1_data, marker2_data marker3_data] = extract_marker_data(filename, dataLines)
%% Input handling

% If dataLines is not specified, define defaults
if nargin < 2
    dataLines = [4, Inf];
end

%%%%%%%%      MARKER 1    %%%%%%%%%%%%
%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 10);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["Var1", "DLC_resnet50_ZebrafishNet2.0Apr28shuffle1_25000", "DLC_resnet50_ZebrafishNet2.0Apr28shuffle1_25000_1", "Var4", "Var5", "Var6", "Var7", "Var8", "Var9", "Var10"];
opts.SelectedVariableNames = ["DLC_resnet50_ZebrafishNet2.0Apr28shuffle1_25000", "DLC_resnet50_ZebrafishNet2.0Apr28shuffle1_25000_1"];
opts.VariableTypes = ["string", "double", "double", "string", "string", "string", "string", "string", "string", "string"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, ["Var1", "Var4", "Var5", "Var6", "Var7", "Var8", "Var9", "Var10"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["Var1", "Var4", "Var5", "Var6", "Var7", "Var8", "Var9", "Var10"], "EmptyFieldRule", "auto");

% Import the data
marker1_data = readtable(filename, opts);

%% Convert to output type
marker1_data = table2array(marker1_data);


%%%%%%%%      MARKER 2    %%%%%%%%%%%%
%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 10);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["Var1", "Var2", "Var3", "Var4", "DLC_resnet50_ZebrafishNet2.0Apr28shuffle1_25000_3", "DLC_resnet50_ZebrafishNet2.0Apr28shuffle1_25000_4", "Var7", "Var8", "Var9", "Var10"];
opts.SelectedVariableNames = ["DLC_resnet50_ZebrafishNet2.0Apr28shuffle1_25000_3", "DLC_resnet50_ZebrafishNet2.0Apr28shuffle1_25000_4"];
opts.VariableTypes = ["string", "string", "string", "string", "double", "double", "string", "string", "string", "string"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, ["Var1", "Var2", "Var3", "Var4", "Var7", "Var8", "Var9", "Var10"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["Var1", "Var2", "Var3", "Var4", "Var7", "Var8", "Var9", "Var10"], "EmptyFieldRule", "auto");

% Import the data
marker2_data = readtable(filename, opts);

%% Convert to output type
marker2_data = table2array(marker2_data);




%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 10);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["Var1", "Var2", "Var3", "Var4", "Var5", "Var6", "Var7", "DLC_resnet50_ZebrafishNet2.0Apr28shuffle1_25000_6", "DLC_resnet50_ZebrafishNet2.0Apr28shuffle1_25000_7", "Var10"];
opts.SelectedVariableNames = ["DLC_resnet50_ZebrafishNet2.0Apr28shuffle1_25000_6", "DLC_resnet50_ZebrafishNet2.0Apr28shuffle1_25000_7"];
opts.VariableTypes = ["string", "string", "string", "string", "string", "string", "string", "double", "double", "string"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, ["Var1", "Var2", "Var3", "Var4", "Var5", "Var6", "Var7", "Var10"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["Var1", "Var2", "Var3", "Var4", "Var5", "Var6", "Var7", "Var10"], "EmptyFieldRule", "auto");

% Import the data
marker3_data = readtable(filename, opts);

%% Convert to output type
marker3_data = table2array(marker3_data);
end