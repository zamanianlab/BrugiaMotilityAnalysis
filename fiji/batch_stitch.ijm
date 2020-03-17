input = getDirectory("Input directory");
//input = "/Volumes/Samsung_T5/20200303/48hr/Plate1-GFP_Plate_67/test/";

output = getDirectory("Output directory");
//output = "/Volumes/Samsung_T5/20200303/48hr/Plate1-GFP_Plate_67/batch_stitch_output/";

list = getFileList(input);

sites_x = getNumber("Input the number of rows per well", 4);
//sites_x = 3;

sites_y = getNumber("Input the number of columns per well", 4);
//sites_y = 3;

sites = sites_x * sites_y;
//sites = 9;

stem = getString("Input the stem of the filenames (everything before the well ID)", "");
//stem = "Plate1-GFP_";

stem_length = lengthOf(stem);

wavelength = getString("Input wavelength that you want to stitch (should be in the form of w1, w2, etc.)", "w1");
//wavelength = "w1";

for (i=0; i < list.length; i = i + sites) {
	if (indexOf(list[i], wavelength) != -1) {
		wellIndexStart = stem_length; // find index of the filename where the 96-well coordinate begins (the number of characters before the stem starts)
    	wellIndexEnd = indexOf(list[i], "_s"); // find index of the filename where the 96-well coordinate ends
    	wellCoord = substring(list[i], wellIndexStart, wellIndexEnd); // create the 96-well coordinate string from the filename
    	genericFilename = replace(list[i], "_s1_", "_s{i}_"); // replace the number with a variable for the stitching plugin
    	action(input, output, genericFilename, list[i], wellCoord); // your function}
	}
    else {

    }
}

function action(input, output, genericFilename, filename, wellCoord) {
  run("Grid/Collection stitching", "type=[Grid: row-by-row] order=[Right & Down                ] grid_size_x=" + sites_x + " grid_size_y=" + sites_y + " tile_overlap=10 first_file_index_i=1 directory=" + input + " file_names=" + genericFilename + " output_textfile_name" + wellCoord + "stitched.txt fusion_method=[Linear Blending] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 computation_parameters=[Save memory (but be slower)] image_output=[Fuse and display]");
    // run("Stack to RGB");
    // run("8-bit"); 
    print(output + filename);
    saveAs("tiff", output + filename);
    close();
}
