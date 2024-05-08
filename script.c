#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define PRINT(...) { printf(__VA_ARGS__); printf("\n"); }
#define FRUIT_AMOUNT 20
#define GENERATION_AMOUNT 100

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef int8_t   i8;
typedef int16_t  i16;
typedef int32_t  i32;
typedef float    f32;
typedef double   f64;

// ---

typedef struct {
  u8    size;
  u8*   layer_sizes;
  f32*  memo;    // One for each node; layer_sizes[0] + ... + layer_sizes[size - 1]
  f32*  biases;  // Same size as memo
  f32*  weights; // layer_sizes[0] * layer_sizes[1] + ... + layer_sizes[size - 2] * layer_sizes[size - 1]
} Network;

typedef struct {
  f32 spikey;
  f32 roughness;
  u8 venomous;
} Fruit;

typedef struct {
  u8 alive;
  Network* network;
} Bot;

Fruit fruits[FRUIT_AMOUNT];

// ---

void init_fruits() {
  for (u8 i = 0; i < FRUIT_AMOUNT; i++) {
    fruits[i].spikey =    (f32) rand() / RAND_MAX;
    fruits[i].roughness = (f32) rand() / RAND_MAX;
    fruits[i].venomous =  fruits[i].spikey > 0.5 && fruits[i].roughness > 0.5;
  }
}

Network* create_nnetwork(u8 num_inn, u8 num_out) {
  u8*  layer_sizes = malloc(sizeof(u8) * 2);
  layer_sizes[0] = num_inn;
  layer_sizes[1] = num_out;
  f32* memo        = malloc((num_inn + num_out) * sizeof(f32));
  f32* biases      = malloc((num_inn + num_out) * sizeof(f32));
  f32* weights     = malloc((num_inn * num_out) * sizeof(f32));
  
  Network* network = malloc(sizeof(Network));
  network->size = 2;
  network->layer_sizes = layer_sizes;
  network->memo        = memo;
  network->biases      = biases;
  network->weights     = weights;

  return network;
}

u8* get_node_weight_amount(Network network) {
  u8 node_amount = 0;
  u8 weight_amount = 0;

  for (u8 i = 0; i < network.size; i++) {
    node_amount += network.layer_sizes[i];

    if (i) weight_amount += network.layer_sizes[i - 1] * network.layer_sizes[i];
  }

  return (u8[]) { node_amount, weight_amount };
}

void randomize_nodes(Network network, f32 min_w, f32 max_w, f32 min_b, f32 max_b) {
  u8* node_weight_amount = get_node_weight_amount(network);
  
  for (u8 i = 0; i < node_weight_amount[0]; i++) network.biases[i]  = ((f32) rand() / RAND_MAX) * (max_b - min_b) + min_b;
  for (u8 i = 0; i < node_weight_amount[1]; i++) network.weights[i] = ((f32) rand() / RAND_MAX) * (max_w - min_w) + min_w;
}

void slightly_randomize_nodes(Network network, f32 var_w, f32 var_b) {
  u8* node_weight_amount = get_node_weight_amount(network);

  for (u8 i = 0; i < node_weight_amount[0]; i++) network.biases[i]  += (((f32) rand() / RAND_MAX) - 0.5) * var_b;
  for (u8 i = 0; i < node_weight_amount[1]; i++) network.weights[i] += (((f32) rand() / RAND_MAX) - 0.5) * var_w;
}

Network* copy_nnetwork(Network _network) {
  u8*  layer_sizes = malloc(sizeof(u8) * 2);
  f32* memo        = malloc((_network.layer_sizes[0] + _network.layer_sizes[1]) * sizeof(f32));
  f32* biases      = malloc((_network.layer_sizes[0] + _network.layer_sizes[1]) * sizeof(f32));
  f32* weights     = malloc((_network.layer_sizes[0] * _network.layer_sizes[1]) * sizeof(f32));

  Network* network = malloc(sizeof(Network));
  network->size = 2;
  network->layer_sizes = layer_sizes;
  network->memo        = memo;
  network->biases      = biases;
  network->weights     = weights;

  u8* node_weight_amount = get_node_weight_amount(_network);
  for (u8 i = 0; i < node_weight_amount[0] + node_weight_amount[1]; i++) network->biases[i] = _network.biases[i];

  return network;
}

u8 get_node_layer(Network network, u8 node_index) {
  for (u8 layer = 0; layer < network.size; layer++) {
    if (node_index < network.layer_sizes[layer]) return layer;
    node_index -= network.layer_sizes[layer];
  }
}

void process_network(Network network) {
  u8* node_weight_amount = get_node_weight_amount(network);
  u8  curr_weight = 0;

  for (u8 node = 0; node < node_weight_amount[0]; node++) {
    network.memo[node] += network.biases[node];

    u8 curr_layer = get_node_layer(network, node);
    if (curr_layer == network.size - 1) continue;

    u8 next_layer_size = network.layer_sizes[curr_layer + 1];
    u8 next_layer_offset = 0;
    for (u8 l = 0; l <= curr_layer; l++) next_layer_offset += network.layer_sizes[l];

    for (u8 forward_node = 0; forward_node < next_layer_size; forward_node++) {
      f32 weight = network.weights[curr_weight++];
      network.memo[next_layer_offset + forward_node] += network.memo[node] * weight;
    }
  }
}

void print_network_dna(Network network) {
  u8* node_weight_amount = get_node_weight_amount(network);
  u8 biases  = node_weight_amount[0];
  u8 weights = node_weight_amount[1];

  for (u8 i = 0; i < biases;  i++) printf("B %f\n", network.biases[i]);
  for (u8 i = 0; i < weights; i++) printf("W %f\n", network.weights[i]);
  printf("\n");
}

// ---

u8 main() {
  srand(time(0));
  init_fruits();
  u8 last_alive_amount = 0;
  u8 best_brain_index = 0;
  u8 best_ate_amount = 0;

  Bot bots[10];
  for (u8 i = 0; i < 10; i++) {
    bots[i].network = create_nnetwork(2, 1);
    randomize_nodes(*bots[i].network, -0.1, 0.1, -0.1, 0.1);
  }

  for (u8 g = 0; g < GENERATION_AMOUNT; g++) {
    u8 alive_amount = 0;

    for (u8 b = 0; b < 10; b++) {
      Network network = *bots[b].network;

      for (u8 f = 0; f < FRUIT_AMOUNT; f++) {
        network.memo[0] = fruits[f].spikey;
        network.memo[1] = fruits[f].roughness;

        process_network(network);

        // Ate
        if (network.memo[2] < 0.5 && fruits[f].venomous) {
          if (f > best_ate_amount) {
            best_brain_index = b;
            best_ate_amount = f;
          }
          break;
        }
        else if (f == FRUIT_AMOUNT - 1) {
          bots[b].alive = 1;

          best_brain_index = b;
          best_ate_amount = f;
        }
      }
    }

    for (u8 b = 0; b < 10; b++) {
      if (bots[b].alive) alive_amount++;

      if (best_brain_index != b) bots[b].network = copy_nnetwork(*bots[best_brain_index].network);
      slightly_randomize_nodes(*bots[b].network, 8, 8);
    }


    if (alive_amount != last_alive_amount || g == GENERATION_AMOUNT || !g) 
      PRINT("%i survived (%iº)", alive_amount, g);

    last_alive_amount = alive_amount;
  }

  return 0;
}
