/* eslint-disable */
// create csv and download it
import * as d3 from "d3";
function convertToCSV(data) {
  const csvContent = data.map(row => row.join(",")).join("\n");
  return encodeURI(csvContent);
}

function downloadCSV(csvContent, fileName) {
  const blob = new Blob([decodeURI(csvContent)], {
    type: "text/csv;charset=utf-8"
  });

  const link = document.createElement("a");
  link.href = window.URL.createObjectURL(blob);
  link.download = fileName;

  document.body.appendChild(link);
  link.click();

  document.body.removeChild(link);
}

// create color csv and download it
const methods = ["Ours-d", "Ours-s", "Color ramp", "TreeColors", "Palettailor"];
function getString(avg_v, std_v) {
  let result = avg_v.toFixed(2) + " - " + std_v.toFixed(2);
  return result;
}

function getLine(num, avg, std) {
  let line = [];
  line.push(methods[num]);
  line.push(
    getString(
      avg["min_perception_difference"],
      std["min_perception_difference"]
    )
  );
  line.push(
    getString(avg["perception_difference"], std["perception_difference"])
  );
  line.push(getString(avg["total_min_dist"], std["total_min_dist"]));
  line.push(getString(avg["min_name_difference"], std["min_name_difference"]));
  line.push(getString(avg["name_difference"], std["name_difference"]));
  line.push(getString(avg["total_name_dist"], std["total_name_dist"]));
  line.push(getString(avg["name_unique"], std["name_unique"]));
  line.push(getString(avg["template_dist"], std["template_dist"]));
  line.push(getString(avg["pp_score"], std["pp_score"]));
  line.push(getString(avg["he_score"], std["he_score"]));
  line.push(getString(avg["time"], std["time"]));
  return line;
}

function saveColorResult(func_get_result, start, end, time) {
  console.assert((end - start) % 32 === 0);
  let result = [];
  result.push([
    "Method",
    "Min dist",
    "Avg dist",
    "Total min dist",
    "Min name dist",
    "Avg name dist",
    "Total min name dist",
    "Name unique",
    "Template dist",
    "PP Score",
    "Harmony value",
    "Time/s"
  ]);

  let [avg, std] = func_get_result(start + 1, start + 11);
  avg["time"] = time[1][0];
  std["time"] = time[1][1];
  result.push(getLine(0, avg, std));

  [avg, std] = func_get_result(start + 11, start + 21);
  avg["time"] = time[2][0];
  std["time"] = time[2][1];
  result.push(getLine(1, avg, std));

  [avg, std] = func_get_result(start + 21, start + 22);
  avg["time"] = time[3];
  std["time"] = 0;
  result.push(getLine(2, avg, std));

  [avg, std] = func_get_result(start, start + 1);
  avg["time"] = time[0];
  std["time"] = 0;
  result.push(getLine(3, avg, std));

  [avg, std] = func_get_result(start + 22, start + 32);
  avg["time"] = time[4][0];
  std["time"] = time[4][1];
  result.push(getLine(4, avg, std));

  // console.log(result);

  let csvContent = convertToCSV(result);
  downloadCSV(csvContent, "color_result.csv");
}

function saveColorResultZoom(func_get_result, parts, times) {
  let result = [];
  result.push([
    "Method",
    "Min dist",
    "Avg dist",
    "Total min dist",
    "Min name dist",
    "Avg name dist",
    "Total min name dist",
    "Name unique",
    "Template dist",
    "PP Score",
    "Harmony value",
    "Time/s"
  ]);

  parts.forEach((level_parts, index) => {
    result.push([
      "level " + index,
      (level_parts[level_parts.length - 1][1] - level_parts[0][0]) / 5
    ]);
    level_parts.forEach((part, part_index) => {
      let [avg, std] = func_get_result(part[0], part[1]);
      let part_times = times.slice(part[0], part[1]);
      avg["time"] = d3.mean(part_times);
      std["time"] = d3.deviation(part_times);
      if (std["time"] === undefined) std["time"] = 0;
      result.push(getLine(part_index, avg, std));
    });
  });

  // console.log(result);

  let csvContent = convertToCSV(result);
  downloadCSV(csvContent, "zoom_result.csv");
}

export { saveColorResult, saveColorResultZoom, convertToCSV, downloadCSV };
