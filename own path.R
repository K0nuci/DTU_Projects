library(visNetwork)
library(binovisualfields)
library(readr)
library(plyr)    
library(writexl)
setwd("~/Desktop/SC")

class <- read_csv("mE808_mol%_data_HKH2_Class.csv")
species <- read_csv("mE808_mol%_data_HKH2_specie.csv")

nodes <- data.frame(id = class$LIPID_OI, label =class$LIPID_OI, values = class$SAMPLE_05, color = c(NA), font.size = 10, shape='circle')
edges <- data.frame(from = c("DAG","DAG","DAG","PI","LPI","PC","LPC","PC","PS","LPS","LPE","PE","PG","LPG",'CL',"PC","PS","PA","PE","PC"), to = c("PC", "TAG", "PA","LPI","PI","LPC","PC","PS","LPS","PS","PE","PS","LPG","PG","PG","PE","PE","DAG","PA","PA"))

unique <- unique(c(edges$from, edges$to))
unused <- setdiff(nodes$id, unique)
nodes$group <- ifelse(nodes$id %in% unused, "unused", nodes$group)

O <- edges[grepl("O-", edges$from) | grepl("O-", edges$to), ]

# Remove the rows from the original 'edges' dataframe
edges <- edges[!(grepl("O-", edges$from) | grepl("O-", edges$to)), ]

'''
#size
nodes$size <- as.numeric(nodes$log2)
for (i in 1:nrow(nodes)) {
  if (nodes$size[i] > 0) {
    nodes$size[i] <- 10 * nodes$size[i]
  } else {
    nodes$size[i] <- 10 / abs(nodes$size[i])
  }
}
'''


add_data <- function(node_1, node_2, edge) {
  # Assuming you have two dataframes: 'nodes' and 'edges'
  # 'nodes' contains columns 'id' and 'label'
  # 'edges' contains columns 'from', 'to'
  
  # Check if 'node_1' and 'node_2' exist in 'nodes'
  if (!(node_1 %in% nodes$id)) {
    # Add row for 'from' node
    from_node <- data.frame(id = node_1, label = node_1)
    nodes <<- rbind.fill(nodes, from_node)
  }
  
  if (!(node_2 %in% nodes$id)) {
    # Add row for 'to' node
    to_node <- data.frame(id = node_2, label = node_2)
    nodes <<- rbind.fill(nodes, to_node)
  }
  
  # Check the 'edge' parameter and add rows accordingly
  if (edge == "from_node_1") {
    new_edge <- data.frame(from = node_1, to = node_2)
    edges <<- rbind.fill(edges, new_edge)
  } else if (edge == "from_node_2") {
    new_edge <- data.frame(from = node_2, to = node_1)
    edges <<- rbind.fill(edges, new_edge)
  } else if (edge == "both_ways") {
    new_edge_1 <- data.frame(from = node_1, to = node_2)
    new_edge_2 <- data.frame(from = node_2, to = node_1)
    edges <<- rbind.fill(edges, new_edge_1, new_edge_2)
  } else {
    stop("Invalid 'edge' parameter. Use 'from_node_1', 'from_node_2', or 'both_ways'.")
  }
  
  # Return the updated dataframes (optional)
  return(list(nodes = nodes, edges = edges))
}

nodes$label <- gsub(" ", "", nodes$label)

# Find the maximum length of names in the 'nodes$label' column
max_name_length <- max(nchar(nodes$label))

# Add spaces at both beginning and end of names to make all names have the same length
nodes$label <- sprintf(paste0("%", max_name_length + 2, "s"), paste0(" ", nodes$label, " "))

# Add spaces to make all names have the same length
nodes$label <- sprintf("%-*s", max_name_length, nodes$label)

nodes$font.size <- 10

# Set the entire 'size' column to have the value "circle"
nodes$shape <- "circle"

nodes$log2 <- log2(nodes$values)

# Assuming 'nodes' is the dataframe containing the 'log2' column
# Calculate the minimum and maximum values of 'log2', excluding NA values
min_value <- min(nodes$log2, na.rm = TRUE)
max_value <- max(nodes$log2, na.rm = TRUE)

# Calculate the interval size (e.g., 0.1)
interval_size <- 0.1

# Calculate intervals for positive and negative values
pos_intervals <- sort(seq(0, max_value, by = interval_size))
neg_intervals <- sort(seq(0, min_value, by = -interval_size))

# Define custom color palettes for positive and negative intervals
colfunc_pos <- colorRampPalette(c("white", "green"))
colfunc_neg <- colorRampPalette(c( "violet", "white"))

# Calculate the number of negative and positive intervals
num_pos_intervals <- length(pos_intervals) - 1
num_neg_intervals <- length(neg_intervals) - 1

pos_col <- colfunc_pos(num_pos_intervals)
neg_col <- colfunc_neg(num_neg_intervals)

for (i in 1:nrow(nodes)) {
  log2_val <- nodes$log2[i]
  
  if (is.na(log2_val)) {
    nodes$color[i] <- "#F88379"  # Set blue color for NA
  } else if (log2_val == 0) {
    nodes$color[i] <- "white"  # Set white color for log2=0
  } else if (log2_val < 0) {
    interval_index <- findInterval(log2_val, neg_intervals)
    interval_index <- pmax(1, pmin(interval_index, num_neg_intervals))  # Ensure interval_index is within valid range
    nodes$color[i] <- neg_col[interval_index]  # Set violet color for intervals log2 < 0
  } else if (log2_val > 0) {
    interval_index <- findInterval(log2_val, pos_intervals)
    interval_index <- pmax(1, pmin(interval_index, num_pos_intervals))  # Ensure interval_index is within valid range
    nodes$color[i] <- pos_col[interval_index]  # Set green color for intervals log2 > 0
  }
}



duplicate_rows <- duplicated(edges)


#hiding nodes_standart
unique_from <- unique(edges$from)
unique_to <- unique(edges$to)
edges_unique <- c(unique_from, unique_to)
nodes$group <- ifelse(!(nodes$id %in% edges_unique), "unused", nodes$group)

#hiding nodes_O-
O_from <- unique(O$from)
O_to <- unique(O$to)
O_unique <- c(O_from, O_to)
nodes$group <- ifelse(!(nodes$id %in% O_unique), "unused_O", nodes$group)

visNetwork(nodes, edges,  main = "Lipids Pathway", submain = list(text = "subtitle",
 style = "font-family:Arial MS;color:#ff0000;font-size:15px;text-align:center;"), 
 footer = "Fig.1 minimal example") %>% 
  visEdges(arrows = "to", color = "black") %>%
  visGroups(groupname = "unused", hidden=TRUE) %>% 
   visConfigure(enabled = TRUE) %>%
   visLayout(randomSeed = 10) %>%
   visOptions(highlightNearest = TRUE, nodesIdSelection = TRUE)

  
network <- visNetwork(nodes, O,  main = "Lipids Pathway O-", submain = list(text = "subtitle",
 style = "font-family:Arial MS;color:#ff0000;font-size:15px;text-align:center;"), 
 footer = "Fig.1 minimal example") %>% 
  visGroups(groupname = "unused_O", hidden=TRUE) %>% 
  visEdges(arrows = "to", color = "black") %>%
   visConfigure(enabled = TRUE) %>%
   visLayout(randomSeed = 10) %>%
   visOptions(highlightNearest = TRUE, nodesIdSelection = TRUE)


visSave(network, file = "network.html", background = "white")



write_xlsx(nodes, "nodes.xlsx")
write_xlsx(edges, "edges.xlsx")
nodes <- nodes[-c(33, 34), ]