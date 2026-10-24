# Fixed direction-propagation controlled training

Keep the current direction propagation research line. TRAIN4, historical DEV2,
seed42,200steps,batch4,identical initial weights and patch sequence. Four arms:
identity propagation, distance-only, learned shared direction, GT-direction
trained/evaluated oracle. All retain auxiliary direction supervision and equal
fusion capacity. Oracle is diagnostic only, broadcast to all candidates, not a
deployable result or mathematical upper bound. Keep radius, losses and original
support bias fixed. Measure total/boundary errors and damage; preserve old data.
