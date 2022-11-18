;; 360 states
;; 3x2
(define (problem impossible)
  (:domain strips-sliding-tile)
  (:objects t1 t2 t3 t4 t5 x1 x2 x3 y1 y2)
  (:init
   (tile t1) (tile t2) (tile t3) (tile t4) (tile t5)
   (position x1) (position x2) (position x3)
   (position y1) (position y2)
   (inc x1 x2) (inc x2 x3) (dec x3 x2) (dec x2 x1)
   (inc y1 y2) (dec y2 y1)
   (blank x1 y2) (at t1 x3 y2) (at t2 x3 y1) (at t3 x1 y1)
   (at t4 x2 y1) (at t5 x2 y2))
  (:goal
   (and (at t1 x2 y1) (at t2 x3 y1) (at t3 x1 y2)
	(at t4 x2 y2) (at t5 x3 y2)))
  )
