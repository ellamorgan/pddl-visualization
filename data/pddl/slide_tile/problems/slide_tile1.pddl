;; 2x2

(define (problem hard1)
  (:domain strips-sliding-tile)
  (:objects t1 t2 t3 x1 x2 y1 y2)
  (:init
   (tile t1) (tile t2) (tile t3)
   (position x1) (position x2)
   (position y1) (position y2)
   (inc x1 x2) (dec x2 x1)
   (inc y1 y2) (dec y2 y1)
   (blank x2 y1) (at t1 x1 y2) (at t2 x2 y2) (at t3 x1 y1))
  (:goal
   (and (at t1 x2 y1) (at t2 x1 y2) (at t3 x2 y2))
  )
)