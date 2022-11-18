;; 22 states
;; 3 blocks
(define (problem small)
(:domain BLOCKS)
(:objects A B C - block)
(:INIT (CLEAR B) (ON B C) (ON C A) (ONTABLE A) (HANDEMPTY))
(:goal (AND (ON C A) (ON A B)))
)